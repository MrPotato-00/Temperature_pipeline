from airflow.decorators import task
from airflow import DAG
from airflow.providers.http.operators.http import HttpOperator
from datetime import datetime
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.empty import EmptyOperator



with DAG(
    dag_id= "dag_trial",
    schedule="@daily",
    start_date= datetime(2025, 1, 1),
    catchup= False
) as dag:
    
    @task
    def createdb():
        postgres_hook= PostgresHook(postgres_conn_id= "postgres_conn")
        create_table_query= '''
            CREATE TABLE IF NOT EXISTS weather_data(
                date DATE PRIMARY KEY,
                temp FLOAT,
                humidity FLOAT,
                wind_speed FLOAT, 
                pressure FLOAT
            );
            '''
        postgres_hook.run(create_table_query)
 
    @task.branch()
    def check_table_emtpy():
        postgres_hook= PostgresHook(postgres_conn_id= "postgres_conn")
        try:
            result = postgres_hook.get_records("SELECT * FROM weather_data order by date desc limit 7;")
            if not result or None in result:
                raise Exception()
            return 'skip_task'
        except Exception as e:
            return "weather_api_last_7days"
            
    
    
    def get_last_7days_data():   
        return HttpOperator(
            task_id= "weather_api_last_7days",
            http_conn_id= "weather_api",
            endpoint= "VisualCrossingWebServices/rest/services/timeline/New%20Delhi/last7days",
            method= "GET",
            data= {
                "key": "{{conn.weather_api.extra_dejson.key}}", 
                "unitGroup": "metric",
                "include": "days"
            },  
            response_filter= lambda response: response.json().get("days", [])
           
        )
           
        
        
    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def process_last_7days_data(response):
    
      
        
        res= (
            datetime.strptime(response["datetime"], "%Y-%m-%d"),
            response["temp"],
            response["humidity"],
            response["windspeed"],
            response["pressure"])

       

        postgres_hook= PostgresHook(postgres_conn_id= "postgres_conn")
        try:
            postgres_hook.run("INSERT INTO weather_data (date, temp, humidity, wind_speed, pressure) VALUES (%s, %s, %s, %s, %s);", parameters= res)
        except Exception as e:
            pass

    def get_today_data_from_api():
    
        result= HttpOperator(
            task_id= "weather_api_today",
            http_conn_id= "weather_api",
            endpoint= "VisualCrossingWebServices/rest/services/timeline/New%20Delhi/today",
            method= "GET",
            data= {
                "key": "{{conn.weather_api.extra_dejson.key}}", 
                "unitGroup": "metric",
                "include": "days"
            },
            response_filter= lambda response: response.json().get("days", []),
            trigger_rule=TriggerRule.NONE_FAILED_OR_SKIPPED
            
        )
        
        return result

    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def process_today_data(response):
        format= "%Y-%m-%d"
        
        data= (
            
            datetime.strptime(response["datetime"], format),
            response["temp"],
            response["humidity"],
            response["windspeed"],
            response["pressure"])
        
        postgres_hook= PostgresHook(postgres_conn_id= "postgres_conn")
        try:
            postgres_hook.run("INSERT INTO weather_data (date, temp, humidity, wind_speed, pressure) VALUES (%s, %s, %s, %s, %s);", parameters= data)
        except Exception as e:
            pass
    @task
    def test_task():
        return "All good"        


    skip_task= EmptyOperator(task_id= "skip_task")
    branch_task= check_table_emtpy()
    last_7days_data= get_last_7days_data()
    today_data= get_today_data_from_api()

    createdb() >> branch_task >> [
        last_7days_data,
        skip_task
    ] 

    [last_7days_data >> process_last_7days_data.expand(response= last_7days_data.output), skip_task] >> today_data >> process_today_data.expand(response= today_data.output)

