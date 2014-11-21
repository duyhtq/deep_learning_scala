package robotbase.brain.layer

import robotbase.brain.function.{Function, Sigmoid}
import robotbase.brain.{C, T}

abstract class Layer(f: Function = Sigmoid) {
  var in: T = null
  var out: T = null
  var x: T = null
  var b: T = null
  var w: T = null
  var db: T = null
  var dw: T = null

  def forward(in: T)

  def backward(loss: T): T

  def update {
    val l2 = w * C.l2Decay
    w = w - ((dw + l2) * (C.learningRate / C.batchSize))
    b = b - (db * (C.learningRate / C.batchSize))
    db.reset
    dw.reset
  }
}

