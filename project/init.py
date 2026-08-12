import os
import json
import requests

from dotenv import load_dotenv


load_dotenv()


GRAFANA_URL = os.getenv("GRAFANA_URL", "http://grafana:3000")

GRAFANA_USER = os.getenv("GRAFANA_ADMIN_USER")
GRAFANA_PASSWORD = os.getenv("GRAFANA_ADMIN_PASSWORD")

PG_HOST = os.getenv("POSTGRES_HOST")
PG_DB = os.getenv("POSTGRES_DB")
PG_USER = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_PORT = os.getenv("POSTGRES_PORT")


def create_api_key():
    auth = (GRAFANA_USER, GRAFANA_PASSWORD)
    headers = {"Content-Type": "application/json"}

    sa_payload = {"name": "ProgrammaticSA", "role": "Admin"}
    response = requests.post(
        f"{GRAFANA_URL}/api/serviceaccounts", auth=auth, headers=headers, json=sa_payload
    )

    if response.status_code == 201:
        sa_id = response.json()["id"]
        print("Service account created")
    elif response.status_code in (409, 400):
        print("Service account already exists, finding it...")
        sa_list = requests.get(f"{GRAFANA_URL}/api/serviceaccounts/search", auth=auth).json()
        sa_id = next(sa["id"] for sa in sa_list["serviceAccounts"] if sa["name"] == "ProgrammaticSA")
    else:
        print(f"Failed to create service account: {response.text}")
        return None

    token_payload = {"name": "ProgrammaticToken"}
    token_response = requests.post(
        f"{GRAFANA_URL}/api/serviceaccounts/{sa_id}/tokens",
        auth=auth,
        headers=headers,
        json=token_payload,
    )

    if token_response.status_code == 200:
        print("Token created successfully")
        return token_response.json()["key"]

    elif "already exists" in token_response.text:
        print("Token already exists, deleting and recreating...")
        tokens = requests.get(
            f"{GRAFANA_URL}/api/serviceaccounts/{sa_id}/tokens", auth=auth
        ).json()
        for token in tokens:
            if token["name"] == "ProgrammaticToken":
                requests.delete(
                    f"{GRAFANA_URL}/api/serviceaccounts/{sa_id}/tokens/{token['id']}",
                    auth=auth,
                )
                print("Old token deleted")
        # try creating again
        retry_response = requests.post(
            f"{GRAFANA_URL}/api/serviceaccounts/{sa_id}/tokens",
            auth=auth,
            headers=headers,
            json=token_payload,
        )
        if retry_response.status_code == 200:
            print("Token created successfully")
            return retry_response.json()["key"]
        else:
            print(f"Failed to create token after retry: {retry_response.text}")
            return None
    else:
        print(f"Failed to create token: {token_response.text}")
        return None
def create_or_update_datasource(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Use a consistent UID for the datasource
    consistent_uid = "vehicle-assistant-postgres"
    
    datasource_payload = {
        "name": "PostgreSQL",
        "type": "grafana-postgresql-datasource",
        "url": "postgres:5432",
        "access": "proxy",
        "user": PG_USER,
        "database": PG_DB,
        "basicAuth": False,
        "isDefault": True,
        "jsonData": {"sslmode": "disable", "postgresVersion": 1300},
        "secureJsonData": {"password": PG_PASSWORD},
        "uid": consistent_uid  # Set consistent UID
    }

    print("Datasource payload:")
    print(json.dumps(datasource_payload, indent=2))

    # First, try to get the existing datasource by UID
    response = requests.get(
        f"{GRAFANA_URL}/api/datasources/uid/{consistent_uid}",
        headers=headers,
    )

    if response.status_code == 200:
        # Datasource exists, update it
        print(f"Updating existing datasource with uid: {consistent_uid}")
        response = requests.put(
            f"{GRAFANA_URL}/api/datasources/uid/{consistent_uid}",
            headers=headers,
            json=datasource_payload,
        )
    else:
        # Datasource doesn't exist, create a new one
        print("Creating new datasource")
        response = requests.post(
            f"{GRAFANA_URL}/api/datasources", headers=headers, json=datasource_payload
        )

    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.text}")

    if response.status_code in [200, 201]:
        print("Datasource created or updated successfully")
        return consistent_uid  # Return the consistent UID
    else:
        print(f"Failed to create or update datasource: {response.text}")
        return None

def create_dashboard(api_key, datasource_uid):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    dashboard_file = dashboard_file = os.path.join(os.path.dirname(__file__), "dashboard.json")

    try:
        with open(dashboard_file, "r") as f:
            dashboard_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: {dashboard_file} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding {dashboard_file}: {str(e)}")
        return

    print("Dashboard JSON loaded successfully.")

    # Update datasource UID in the dashboard JSON
    panels_updated = 0
    for panel in dashboard_json.get("panels", []):
        # Update panel-level datasource
        if isinstance(panel.get("datasource"), dict):
            panel["datasource"]["uid"] = datasource_uid
            panels_updated += 1
        
        # Update target-level datasources
        if isinstance(panel.get("targets"), list):
            for target in panel["targets"]:
                if isinstance(target.get("datasource"), dict):
                    target["datasource"]["uid"] = datasource_uid
                    panels_updated += 1

    print(f"Updated datasource UID for {panels_updated} panels/targets.")

    # Remove keys that shouldn't be included when creating a new dashboard
    dashboard_json.pop("id", None)
    dashboard_json.pop("uid", None)
    dashboard_json.pop("version", None)

    # Prepare the payload
    dashboard_payload = {
        "dashboard": dashboard_json,
        "overwrite": True,
        "message": "Updated by Python script",
    }

    print("Sending dashboard creation request...")

    response = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db", headers=headers, json=dashboard_payload
    )

    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.text}")

    if response.status_code == 200:
        print("Dashboard created successfully")
        return response.json().get("uid")
    else:
        print(f"Failed to create dashboard: {response.text}")
        return None


def main():
    api_key = create_api_key()
    if not api_key:
        print("API key creation failed")
        return

    datasource_uid = create_or_update_datasource(api_key)
    if not datasource_uid:
        print("Datasource creation failed")
        return

    create_dashboard(api_key, datasource_uid)


if __name__ == "__main__":
    main()