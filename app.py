from flask import Flask, request, jsonify
import joblib
import numpy as np

# Cargar el modelo previamente entrenado
model = joblib.load('modelo_columna.bin')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = ['incidencia_pelvica', 'inclinacion_pelvica',
                    'angulo_lordosis_lumbar', 'pendiente_sacra', 
                    'radio_pelvico', 'grado_espondilolistesis']
        
        # Verificar que todos los features están en el JSON recibido
        if not all(feature in data for feature in features):
            return jsonify({'error': 'Faltan características en la entrada'}), 400
        
        # Convertir a numpy array
        input_data = np.array([data[feature] for feature in features]).reshape(1, -1)
        
        # Realizar la predicción
        prediction = model.predict(input_data)
        
        # Retornar el resultado
        return jsonify({'prediccion': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
