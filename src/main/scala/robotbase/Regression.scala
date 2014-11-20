package robotbase

class Regression extends Layer {

  def forward(in: Tensor) = {
    this.in = in
    out = in
  }

  def backward(y: Tensor): Tensor = o.-(out, y)

}
