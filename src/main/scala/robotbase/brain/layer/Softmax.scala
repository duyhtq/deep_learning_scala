package robotbase.brain.layer

import robotbase.brain.T

class Softmax extends Layer {

  def forward(in: T) {
    this.in = in
    val e = in.values.map(xi => Math.exp(xi - in.values.max))
    val sum = e.reduce(_ + _)
    out = new T(e.map(ei => ei / sum), in.dim)
  }

  def backward(y: T): T = out - y

}
