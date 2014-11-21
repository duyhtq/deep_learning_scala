package robotbase.brain.layer

import robotbase.brain.function.Function
import robotbase.brain.{R, T}

class Full(f: Function, m: Int, n: Int) extends Layer(f) {

  x = new T(n)
  b = new T(R.random, n)
  w = new T(R.random, m, n)
  db = new T(n)
  dw = new T(m, n)

  def forward(in: T) {
    this.in = in
    x = (in * w) + b
    out = f(x)
  }

  def backward(loss: T): T = {
    val gradient = f.d(x) *+ loss
    db = db + gradient
    dw = dw + (in x gradient)
    gradient * w.tranpose
  }

}