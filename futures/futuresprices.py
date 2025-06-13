import requests

API_KEY = "nuDX73vj2483KSUgvenkj9t50oA0vgvA4WcuRAER"
BASE_URL = "https://api.mae.com.ar/MarketData/v1/mercado/cotizaciones/derivados"

headers = {
    "x-api-key": API_KEY
}

def obtener_todo():
    response = requests.get(BASE_URL, headers=headers)
    print("Status code:", response.status_code)
    print("Headers:", response.headers)
    print("Respuesta:", response.text)

if __name__ == "__main__":
    obtener_todo()