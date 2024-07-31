from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_KEY')


#examples = [
 #   {"document1": """Human \n Show KPI Dashboard \n AI \n {"action" : "open", "link": "kpi-dashboard"} """},
  #  {"document2": """Human \n Edit KPI Dashboard for last month for VOA. \n AI \n {"action" : "update", "link": "kpi-dashboard", "start_date": "DD/MM/YYYY", "end_date": "DD/MM/YYYY"} """},
   # {"document3": """Human \n Show KPI Dashboard for VOA from April 1st 2024 to April 7th 2024. \n AI \n {"action" : "open", "link": "kpi-dashboard", "partner": "VOA", "start_date": "01/04/2024", "end_date": "07/04/2024"}} """}
#]

examples = [
    {"input": "Show KPI Dashboard.", "output": """{"action" : "open", "link": "kpi-dashboard"}"""},
    {"input": "Edit KPI Dashboard for last month for VOA.", "output": """{"action" : "update", "link": "kpi-dashboard", "start_date": "DD/MM/YYYY", "end_date": "DD/MM/YYYY"}"""},
    {"input": "Show KPI Dashboard for partner VOA.", "output": """{"action" : "open", "link": "kpi-dashboard", "partner": "VOA"}"""},
    {"input": "Show KPI Dashboard for VOA from April 1st 2024 to April 7th 2024.", "output": """{"action" : "open", "link": "kpi-dashboard", "partner": "VOA", "start_date": "01/04/2024", "end_date": "07/04/2024"}}"""},
    {"input": "Show Leadership Dashboard.", "output": """{"action" : "open", "link": "leadership-dashboard"}"""},
    {"input": "Edit Leadership Dashboard for last week for VOA.", "output": """{"action" : "update", "link": "leadership-dashboard", "start_date": "DD/MM/YYYY", "end_date": "DD/MM/YYYY"}"""},
    {"input": "Show Leadership Dashboard for provider VOA.", "output": """{"action" : "open", "link": "leadership-dashboard", "provider": "VOA"}"""},
    {"input": "Show Registry report with member id as 100263. ", "output": """{"action" : "open", "link": "registry", "member_id": "100263"}"""},
    {"input": "Show Configuration screen.","output": """{"action": "open", "link": "configuration"}"""},
    {"input": "Open Configuration screen.","output": """{"action": "open", "link": "configuration"}"""},
    {"input": "Show User Management screen.","output": """{"action": "open", "link": "user-mgmt"}"""},
    {"input": "Open User Management screen.","output": """{"action": "open", "link": "user-mgmt"}"""},
    {"input": "Create a new user.","output": """{"action": "create", "link": "user-mgmt"}"""},
    {"input": "Show Data Manager screen.","output": """{"action": "open", "link": "data-mngr"}"""},
    {"input": "Open Data Manager screen.","output": """{"action": "open", "link": "data-mngr"}"""},
    {"input": "Update all data from EHR.","output": """{"action": "update", "link": "data-mngr"}"""},
    {"input": "Show Onboarding Report","output": "{\"action\": \"open\", \"link\": \"onboarding\"}"},
    {"input": "Show Onboarding Report with Current Member Status as New","output": "{\"action\": \"open\", \"link\": \"onboarding\", \"Current Member Status\": \"New\"}"},
    {"input": "Show Onboarding report where created date / member created date is greater than 05th March","output": "{\"action\": \"open\", \"link\": \"onboarding\", \"start_date\": \"05/03/24\", \"end_date\": \"02/05/24\"}"},
    {"input": "Show Onboarding report where due date for contact is less than a week","output": "{\"action\": \"open\", \"link\": \"onboarding\", \"start_date\": \"01/05/2024\", \"end_date\": \"08/05/2024\"}"},
    {"input": "Show Onboarding report with last member outreach date in last 10 days","output": "{\"action\": \"open\", \"link\": \"onboarding\", \"start_date\": \"21/04/2024\", \"end_date\": \"01/05/2024\"}"},
    {"input":"Show Onboarding report with ops as me","output":"link: onboarding , OPS: me"},
    {"input": "Show Quality Report","output": "{\"action\": \"open\", \"link\": \"quality\"}"},
    {"input": "Open Quality Report","output": "{\"action\": \"open\", \"link\": \"quality\"}"},
    {"input": "Show Billing Report with partner/customer as VOA and month as January","output": "{\"action\": \"open\", \"link\": \"billing\", \"partner\": \"VOA\", \"start_date\": \"01/01/2024\", \"end_date\": \"31/01/2024\"}"},
    {"input": "Open Billing Report with partner/customer as VOA and month as January","output": "{\"action\": \"open\", \"link\": \"billing\", \"partner\": \"VOA\", \"start_date\": \"01/01/2024\", \"end_date\": \"31/01/2024\"}"},
    {"input": "Show Billing Report for last month for VOA.","output": "{\"action\": \"open\", \"link\": \"billing\", \"partner\": \"VOA\", \"start_date\": \"01/04/2024\", \"end_date\": \"30/04/2024\"}"},
    {"input": "Open Billing Report for last month for VOA.","output": "{\"action\": \"open\", \"link\": \"billing\", \"partner\": \"VOA\", \"start_date\": \"01/04/2024\", \"end_date\": \"01/14/2024\"}"},
    {"input": "Initiate Billing Report for January 2024.","output": "{\"action\": \"open\", \"link\": \"billing\", \"partner\": \"VOA\", \"start_date\": \"01/01/2024\", \"end_date\": \"31/01/2024\"}"},
    {"input": "Open Billing Report for January 2024.","output": "{\"action\": \"open\", \"link\": \"billing\", \"partner\": \"VOA\", \"start_date\": \"01/01/2024\", \"end_date\": \"31/01/2024\"}"},
    {"input": "Show Clinical Summary Report with partner/customer as VOA and month as January","output": "{\"action\": \"open\", \"link\": \"clinical-summary\", \"partner\": \"VOA\", \"start_date\": \"01/01/2024\", \"end_date\": \"31/01/2024\"}"},
    {"input": "Open Clinical Summary Report with partner/customer as VOA and month as January","output": "{\"action\": \"open\", \"link\": \"clinical-summary\", \"partner\": \"VOA\", \"start_date\": \"01/01/2024\", \"end_date\": \"31/01/2024\"}"},
    {"input": "Show Clinical Summary Report for Jan for VOA","output": "{\"action\": \"open\", \"link\": \"clinical-summary\", \"partner\": \"VOA\", \"start_date\": \"01/01/2024\", \"end_date\": \"31/01/2024\"}"},
    {"input": "Open Clinical Summary Report for Jan for VOA","output": "{\"action\": \"open\", \"link\": \"clinical-summary\", \"partner\": \"VOA\", \"start_date\": \"01/01/2024\", \"end_date\": \"31/01/2024\"}"},
    {"input": "Show scheduled report with coach as Emily Clark","output": "{\"action\": \"open\", \"link\": \"enrollment\", \"coach\": \"Emily Clark\"}"},
    {"input": "Show scheduled report where OPS is John Smith","output": "{\"action\": \"open\", \"link\": \"enrollment\", \"OPS\": \"John Smith\"}"},
    {"input": "Show scheduled report where BHCM is Jane Doe","output": "{\"action\": \"open\", \"link\": \"enrollment\", \"BHCM\": \"Jane Doe\"}"},
    {"input": "Show Scheduled Report for me","output": "{\"action\": \"open\", \"link\": \"enrollment\", \"user\": \"logged_in_user\"}"},
    {"input": "Show Scheduled Report","output": "{\"action\": \"open\", \"link\": \"enrollment\"}"},
    {"input": "Show BHCM Specialist Communication Report","output": "{\"action\": \"open\", \"link\": \"bhcm-specialist-communication\"}"},
    {"input": "Show BHCM Specialist Communication Report for member id 12345","output": "{\"action\": \"open\", \"link\": \"bhcm-specialist-communication\", \"member_id\": \"12345\"}"},
    {"input": "Open BHCM Specialist Communication Report","output": "{\"action\": \"open\", \"link\": \"bhcm-specialist-communication\"}"},
    {"input": "Open BHCM Specialist Communication Report for member id 12345","output": "{\"action\": \"open\", \"link\": \"bhcm-specialist-communication\", \"member_id\": \"12345\"}"},
    {"input": "Show Open Evaluations Report","output": "{\"action\": \"open\", \"link\": \"open-evaluations\"}"},
    {"input": "Show evaluation report with member id 67890","output": "{\"action\": \"open\", \"link\": \"open-evaluations\", \"member_id\": \"67890\"}"},
    {"input": "Open Open Evaluations Report","output": "{\"action\": \"open\", \"link\": \"open-evaluations\"}"},
    {"input": "Open evaluation report with member id 67890","output": "{\"action\": \"open\", \"link\": \"open-evaluations\", \"member_id\": \"67890\"}"}
]

for i in examples:
    i['input'] = "Human \n " + i['input'] + " \n"
    i['output'] = "AI \n " + i['output'] + " \n"

to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples, persist_directory="./chroma_db")


