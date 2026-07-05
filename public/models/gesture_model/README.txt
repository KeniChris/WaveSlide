Aquí debes poner los archivos convertidos de tu modelo TensorFlow.js:

- model.json
- group1-shard1ofX.bin
- group1-shard2ofX.bin
- etc.

La app carga el modelo desde:
/public/models/gesture_model/model.json

El orden de clases usado en src/app.js es:
['call', 'fist', 'like', 'two_up']

Si tu entrenamiento usó otro orden, cambia CLASS_NAMES en src/app.js.
