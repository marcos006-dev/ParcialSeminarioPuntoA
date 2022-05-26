import * as tf from '@tensorflow/tfjs';
const inputValorCalulcar = document.getElementById('valorACalcular');
const contenedorResultado = document.getElementById('resultado');
const mostrarVisor = document.getElementById('mostrarVisor');
let modeloEntrenado;

// se establece la constante para el visor
// const surface = tfvis.visor().surface({
//   name: 'Estado del entrenamiento del modelo',
//   tab: 'Entrenamiento',
// });
// funcion para calcular los valores de Y
const calcularValoresY = (paramValoresX) => {
  const arrayResultadosY = [];
  for (let i = 0; i < paramValoresX.length; i++) {
    const y = 2 * paramValoresX[i] + 3;
    arrayResultadosY.push(y);
  }
  return arrayResultadosY;
};

// funcion para cargar la grafica
const funcionLineal = async () => {
  contenedorResultado.innerHTML = 'El modelo se esta entrenando...';

  // valores de X
  const valoresInicialesX = [-1, 0, 1, 2, 3, 4];

  // se calculan los valores de Y
  const resultadoY = calcularValoresY(valoresInicialesX);

  // se arman los tensores para el modelo
  const xs = tf.tensor2d(valoresInicialesX, [6, 1]);
  const ys = tf.tensor2d(resultadoY, [6, 1]);

  // se crea el modelo
  const model = tf.sequential();

  // se crea la capa de entrada
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // se define el optimizador
  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd',
    metrics: ['accuracy'],
  });

  // se entrena el modelo
  await model.fit(xs, ys, {
    epochs: 250,
    callbacks: [
      // tfvis.show.fitCallbacks(surface, ['loss', 'acc'], {
      //   name: 'Entrenamiento',
      // }),
      {
        onEpochEnd: async (epoch, logs) => {
          console.log('Epoch:' + epoch + ' Loss:' + logs.loss);
        },
      },
    ],
  });

  inputValorCalulcar.disabled = false;
  inputValorCalulcar.focus();

  modeloEntrenado = model;
  contenedorResultado.innerHTML = 'Modelo entrenado, listo para usar';
};

// funcion que se ejecuta al cargar por completo la pagina
document.addEventListener('DOMContentLoaded', () => {
  // funcion que carga el modelo
  funcionLineal();

  // funcion que se ejecuta al presionar el boton calcular
  inputValorCalulcar.addEventListener('keyup', (event) => {
    if (event.keyCode === 13) {
      event.preventDefault();
      // convertir a numero el valor ingresado
      const valorACalcular = parseInt(inputValorCalulcar.value);
      // se realiza la prediccion
      const resultado = modeloEntrenado.predict(
        tf.tensor2d([valorACalcular], [1, 1])
      );

      // se obtiene el valor de la prediccion
      const valorResultado = resultado.dataSync();
      // console.log(valorResultado);
      // se muestra el resultado en la grafica
      armarGrafica(valorACalcular, valorResultado[0]);
      contenedorResultado.innerHTML = `El resultado aproximado para Y es de: ${valorResultado}`;
    }
  });

  // funcion para mostrar y ocultar el visor
  mostrarVisor.addEventListener('click', () => {
    tfvis.visor().toggle();
  });
});

// funcion para armar la grafica
const armarGrafica = (valorX, valorY) => {
  // const trace1 =

  // Define Data
  const data = [
    {
      x: [valorX],
      y: [valorY],
      mode: 'markers',
      // type: 'scatter',
    },
  ];

  // Define Layout
  const layout = {
    xaxis: { range: [Math.abs(valorX), valorX], title: 'Valores de X' },
    yaxis: { range: [Math.abs(valorY), valorY], title: 'Valores de Y' },
    // title: 'House Prices vs. Size',
  };

  Plotly.newPlot('myPlot', data, layout);
};
