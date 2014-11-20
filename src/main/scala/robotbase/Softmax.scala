package robotbase

class Softmax extends Layer {

  def forward(in: Tensor) = {
    this.in = in
    val e = in.a.map(xi => Math.exp(xi - in.a.max))
    val sum = e.reduce(_ + _)
    out = new Tensor(e.map(ei => ei / sum), in.dim)
  }

  def backward(y: Tensor): Tensor = o.-(out, y)

}
