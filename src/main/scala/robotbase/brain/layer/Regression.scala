package robotbase.brain.layer

import robotbase.brain.T

class Regression extends Layer {

  def forward(in: T) {
    this.in = in
    out = in
  }

  def backward(y: T): T = out - y

}
