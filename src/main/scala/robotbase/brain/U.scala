package robotbase.brain

import java.util.List

import org.apache.log4j.Logger

object U {

  def fromJavaListToScalaArray(a: List[Integer]) = {
    val b = new Array[Int](a.size)
    for (i <- 0 until b.length)
      b.update(i, a.get(i))
    b
  }

  def log(s: String) = Logger.getLogger("net").info(s)
}
