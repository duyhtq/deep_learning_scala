package robotbase

class Full(f: o.function, m: Int, n: Int) extends Layer(f) {

  x = new Tensor(n)
  b = new Tensor(R.random, n)
  w = new Tensor(R.random, m, n)
  db = new Tensor(n)
  dw = new Tensor(m, n)

  def forward(in: Tensor) = {
    this.in = in
    x = o.+(o.*(in, w), b)
    out = f(x)
  }

  def backward(loss: Tensor): Tensor = {
    val gradient = o.*+(f.d(x), loss)
    db = o.+(db, gradient)
    dw = o.+(dw, o.x(in, gradient))
    o.*(gradient, o.t(w))
  }

}