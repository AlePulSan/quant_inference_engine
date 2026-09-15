import csv
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.data.live import StockDataStream

# 1. Configura tus credenciales (Paper Trading)
API_KEY = "PKKYIYAL3DTYGQJIQCVQEIMNXP"
SECRET_KEY = "516dPiYsuCDmf7V5HWSn2bu31ABHQwZiCGPFJmhpCs5C"
SYMBOL = "SPY" 
CSV_FILE = "data/datos_mercado.csv" # Guardamos directamente en la carpeta data

# 2. Inicializa el archivo
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp', 'symbol', 'price', 'size'])

# 3. Define el cliente WebSocket para Crypto
#stream = CryptoDataStream(API_KEY, SECRET_KEY)
stream = StockDataStream(API_KEY, SECRET_KEY)

async def procesar_trade(data):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([data.timestamp, data.symbol, data.price, data.size])
    
    print(f"[{data.timestamp}] {data.symbol} | Precio: ${data.price} | Volumen: {data.size}")

# 4. Suscripción y ejecución
stream.subscribe_trades(procesar_trade, SYMBOL)

print(f"Escuchando trades en vivo de {SYMBOL}... Presiona Ctrl+C para detener.")
stream.run()