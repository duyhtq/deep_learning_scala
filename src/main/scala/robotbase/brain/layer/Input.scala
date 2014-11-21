package robotbase.brain.layer

import java.util.List

import robotbase.brain.{U, T}

class Input(dim: Array[Int]) extends Layer() {

  def this(d: Int*) = this(d.toArray)

  def forward(in: T) = {
    // take an input (usually a vector) and reshape it into n-dim
    this.in = in
    out = in.copy
    out.updateDim(dim)
  }

  def backward(loss: T): T = {
    // no back prop at the input layer
    null
  }

}