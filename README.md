# WaveSlide Gesture PDF

App web básica para controlar un PDF con gestos detectados por webcam.

Flujo:

1. MediaPipe Hands detecta la mano.
2. Se calcula un bbox usando los landmarks.
3. Se recorta la mano a 224x224.
4. MobileNetV2 convertido a TensorFlow.js clasifica el gesto.
5. El gesto controla el PDF.

## Instalación

```bash
npm install
npm run dev
```

Abre la URL que muestra Vite, normalmente:

```text
http://127.0.0.1:5173
```

No abras `index.html` con doble clic porque la cámara y los modelos necesitan ejecutarse desde servidor local.

## Modelo

Coloca tu modelo convertido en:

```text
public/models/gesture_model/model.json
public/models/gesture_model/*.bin
```

## Clases

El orden actual en `src/app.js` es:

```js
['call', 'fist', 'like', 'two_up']
```

Debe coincidir con el orden usado en entrenamiento.
