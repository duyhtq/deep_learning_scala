package robotbase

abstract class Layer(val f: o.function = o.sigmoid) {
  var in: Tensor = null
  var out: Tensor = null
  var x: Tensor = null
  var b: Tensor = null
  var w: Tensor = null
  var db: Tensor = null
  var dw: Tensor = null

  def forward(in: Tensor)

  def backward(loss: Tensor): Tensor

  def update = {
    val l2 = o.*(C.l2Decay, w)
    w = o.-(w, o.*(C.learningRate / C.batchSize, o.+(dw, l2)))
    b = o.-(b, o.*(C.learningRate / C.batchSize, db))
    db.reset
    dw.reset
  }
}