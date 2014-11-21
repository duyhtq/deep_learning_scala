package robotbase.brain

import com.typesafe.config.Config

object C {
  var nClass = 0
  var normalizationFactor = 0
  var nIteration = 0
  var learningRate = 0.0
  var l1Decay = 0.0
  var l2Decay = 0.0
  var reportSize = 0
  var batchSize = 0
  var trainSize = 0
  var testSize = 0

  def update(c: Config) {
    nClass = c.getInt("param.nClass")
    normalizationFactor = c.getInt("param.normalizationFactor")
    nIteration = c.getInt("param.nIteration")
    learningRate = c.getDouble("param.learningRate")
    l1Decay = c.getDouble("param.l1Decay")
    l2Decay = c.getDouble("param.l2Decay")
    reportSize = c.getInt("param.reportSize")
    batchSize = c.getInt("param.batchSize")
    trainSize = c.getInt("param.trainSize")
    testSize = c.getInt("param.testSize")
  }
}
