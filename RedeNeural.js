function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(x) {
  return x * (1 - x);
}

class RedeNeural {
  constructor(i_nodes, h1_nodes, h2_nodes, h3_nodes, o_nodes) {
    this.i_nodes = i_nodes;
    this.h1_nodes = h1_nodes;
    this.h2_nodes = h2_nodes;
    this.h3_nodes = h3_nodes;
    this.o_nodes = o_nodes;

    this.bias_ih1 = new Matrix(this.h1_nodes, 1);
    this.bias_ih1.randomize();
    this.bias_h1h2 = new Matrix(this.h2_nodes, 1);
    this.bias_h1h2.randomize();
    this.bias_h2h3 = new Matrix(this.h3_nodes, 1);
    this.bias_h2h3.randomize();
    this.bias_h3o = new Matrix(this.o_nodes, 1);
    this.bias_h3o.randomize();

    this.weigths_ih1 = new Matrix(this.h1_nodes, this.i_nodes);
    this.weigths_ih1.randomize();

    this.weigths_h1h2 = new Matrix(this.h2_nodes, this.h1_nodes);
    this.weigths_h1h2.randomize();

    this.weigths_h2h3 = new Matrix(this.h3_nodes, this.h2_nodes);
    this.weigths_h2h3.randomize();

    this.weigths_h3o = new Matrix(this.o_nodes, this.h3_nodes);
    this.weigths_h3o.randomize();

    this.learning_rate = 0.1;
  }

  train(arr, target) {
    // INPUT -> HIDDEN 1

    let input = Matrix.arrayToMatrix(arr);
    let hidden1 = Matrix.multiply(this.weigths_ih1, input);

    hidden1 = Matrix.add(hidden1, this.bias_ih1);

    hidden1.map(sigmoid);

    // HIDDEN 1 -> HIDDEN 2

    let hidden2 = Matrix.multiply(this.weigths_h1h2, hidden1);

    hidden2 = Matrix.add(hidden2, this.bias_h1h2);

    hidden2.map(sigmoid);

    // HIDDEN 2 -> HIDDEN 3

    let hidden3 = Matrix.multiply(this.weigths_h2h3, hidden2);

    hidden3 = Matrix.add(hidden3, this.bias_h2h3);

    hidden3.map(sigmoid);

    // HIDDEN -> OUTPUT

    let output = Matrix.multiply(this.weigths_h3o, hidden3);
    output = Matrix.add(output, this.bias_h3o);

    output.map(sigmoid);

    // BACKPROPAGATION

    // OUTPUT -> HIDDEN 3

    let expected = Matrix.arrayToMatrix(target);

    let output_error = Matrix.subtract(expected, output);

    let d_output = Matrix.map(output, dsigmoid);

    let hidden3_T = Matrix.transpose(hidden3);

    let gradient = Matrix.hadamard(d_output, output_error);
    gradient = Matrix.escalar_multiply(gradient, this.learning_rate);

    // Adjust Bias O -> H 3

    this.bias_h3o = Matrix.add(this.bias_h3o, gradient);

    // Adjust Weigths O -> H 3

    let weigths_h3o_deltas = Matrix.multiply(gradient, hidden3_T);

    this.weigths_h3o = Matrix.add(this.weigths_h3o, weigths_h3o_deltas);

    // HIDDEN 3 -> HIDDEN 2

    let weigths_h3o_T = Matrix.transpose(this.weigths_h3o);

    let hidden3_error = Matrix.multiply(weigths_h3o_T, output_error);

    let d_hidden3 = Matrix.map(hidden3, dsigmoid);

    let hidden2_T = Matrix.transpose(hidden2);

    let gradient_H3 = Matrix.hadamard(d_hidden3, hidden3_error);
    gradient_H3 = Matrix.escalar_multiply(gradient_H3, this.learning_rate);

    // Adjust Bias O -> H 3

    this.bias_h2h3 = Matrix.add(this.bias_h2h3, gradient_H3);

    // Adjust Weigths H 3 -> H 2

    let weigths_h2h3_deltas = Matrix.multiply(gradient_H3, hidden2_T);

    this.weigths_h2h3 = Matrix.add(this.weigths_h2h3, weigths_h2h3_deltas);

    // HIDDEN 2 -> HIDDEN 1

    let weigths_h2h3_T = Matrix.transpose(this.weigths_h2h3);

    let hidden2_error = Matrix.multiply(weigths_h2h3_T, hidden3_error);

    let d_hidden2 = Matrix.map(hidden2, dsigmoid);

    let hidden1_T = Matrix.transpose(hidden1);

    let gradient_H2 = Matrix.hadamard(d_hidden2, hidden2_error);
    gradient_H2 = Matrix.escalar_multiply(gradient_H2, this.learning_rate);

    // Adjust Bias O -> H 3

    this.bias_h1h2 = Matrix.add(this.bias_h1h2, gradient_H2);

    // Adjust Weigths H 3 -> H 2

    let weigths_h1h2_deltas = Matrix.multiply(gradient_H2, hidden1_T);

    this.weigths_h1h2 = Matrix.add(this.weigths_h1h2, weigths_h1h2_deltas);

    // HIDDEN 1 -> INPUT

    let weigths_h1h2_T = Matrix.transpose(this.weigths_h1h2);

    let hidden1_error = Matrix.multiply(weigths_h1h2_T, hidden2_error);

    let d_hidden1 = Matrix.map(hidden1, dsigmoid);

    let input_T = Matrix.transpose(input);

    let gradient_H1 = Matrix.hadamard(d_hidden1, hidden1_error);
    gradient_H1 = Matrix.escalar_multiply(gradient_H1, this.learning_rate);

    // Adjust Bias O->H

    this.bias_ih1 = Matrix.add(this.bias_ih1, gradient_H1);

    // Adjust Weigths H->I

    let weigths_ih1_deltas = Matrix.multiply(gradient_H1, input_T);

    this.weigths_ih1 = Matrix.add(this.weigths_ih1, weigths_ih1_deltas);
  }

  predict(arr) {
    // INPUT -> HIDDEN 1

    let input = Matrix.arrayToMatrix(arr);

    let hidden1 = Matrix.multiply(this.weigths_ih1, input);
    hidden1 = Matrix.add(hidden1, this.bias_ih1);

    hidden1.map(sigmoid);

    // HIDDEN 1 -> HIDDEN 2

    let hidden2 = Matrix.multiply(this.weigths_h1h2, hidden1);
    hidden2 = Matrix.add(hidden2, this.bias_h1h2);

    hidden2.map(sigmoid)

    // HIDDEN 2 -> HIDDEN 3

    let hidden3 = Matrix.multiply(this.weigths_h2h3, hidden2);
    hidden3 = Matrix.add(hidden3, this.bias_h2h3);

    hidden3.map(sigmoid)

    // HIDDEN -> OUTPUT
    let output = Matrix.multiply(this.weigths_h3o, hidden3);
    output = Matrix.add(output, this.bias_h3o);

    output.map(sigmoid);
    
    output = Matrix.MatrixToArray(output);

    return output;
  }
}
