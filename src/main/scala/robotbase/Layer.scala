package rb.ai

import org.apache.spark.mllib.linalg.{Matrices, Vectors}

case class Layer(val l1: Int, val l2: Int) {
  var x = f.randomVector(l2)
  var a = f.randomVector(l2)
  var b = f.randomVector(l2)
  var w = f.randomMatrix(l1, l2)
  var db = Vectors.zeros(l2)
  var dw = Matrices.dense(l1, l2, Vectors.zeros(l1 * l2).toArray)

}
