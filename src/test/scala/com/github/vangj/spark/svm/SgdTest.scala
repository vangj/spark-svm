package com.github.vangj.spark.svm

import com.github.vangj.spark.svm.Sgd.{DataPoint, Params, Vector}
import com.holdenkarau.spark.testing.SharedSparkContext
import org.apache.log4j.LogManager
import org.scalatest.{FlatSpec, Matchers}

object SgdTest {
  @transient lazy val logger = LogManager.getLogger(SgdTest.getClass)

  private def print(raw: List[DataPoint], w: Vector, b: Double): Unit = {
    raw.foreach(p => {
      val r = p.x * w - b
      val r_formatted = "%.5f".format(r)
      logger.debug(s"${p.x} * ${w} - ${b} = ${r_formatted}")
    })
  }
}

class SgdTest extends FlatSpec with Matchers with SharedSparkContext {

  "super duper simple SVM learning" should "work" in {
    val raw = List(
      new DataPoint(-1.0d, new Vector(List(0.0d, 0.0d))),
      new DataPoint(1.0d, new Vector(List(1.0d, 1.0d)))
    )

    val data = sc.parallelize(raw)
    val params = new Params(200, 2, .01, 37L)
    val r = Sgd.learnHyperplane(data, params)
    val b = r._1
    val w = r._2

    raw.foreach(p => {
      val r = p.x * w - b
      if (p.y < 1) {
        (r < 0) should be(true)
      } else {
        (r > 0) should be(true)
      }
    })

    SgdTest.print(raw, w, b)
  }

  "super simple SVM learning" should "work" in {
    val raw = List(
      new DataPoint(-1.0d, new Vector(List(1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d)))
    )

    val data = sc.parallelize(raw)
    val params = new Params(200, 20, .01, 37L)
    val r = Sgd.learnHyperplane(data, params)
    val b = r._1
    val w = r._2

    raw.foreach(p => {
      val r = p.x * w - b
      if (p.y < 1) {
        (r < 0) should be(true)
      } else {
        (r > 0) should be(true)
      }
    })

    SgdTest.print(raw, w, b)
  }

  "simple SVM learning" should "work" in {
    val raw = List(
      new DataPoint(-1.0d, new Vector(List(1.0d, 1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d, 1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d, 1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d, 1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d, 1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d, 1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d, 1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d, 1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d, 1.0d))),
      new DataPoint(-1.0d, new Vector(List(1.0d, 1.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d, 10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d, 10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d, 10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d, 10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d, 10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d, 10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d, 10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d, 10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d, 10.0d))),
      new DataPoint(1.0d, new Vector(List(10.0d, 10.0d)))
    )

    val data = sc.parallelize(raw)
    val params = new Params(400, 100, 0.01d, 37L)
    val r = Sgd.learnHyperplane(data, params)
    val b = r._1
    val w = r._2

    raw.foreach(p => {
      val x1 = (p.x * w)
      val x2 = x1 - b
      val r = (p.x * w) - b
      if (p.y < 1) {
        (r < 0) should be(true)
      } else {
        (r > 0) should be(true)
      }
    })

    SgdTest.print(raw, w, b)
  }

  "learning from iris" should "work" in {
    Sgd.main(Array(
      "--T", "400",
      "--k", "50",
      "--lambda", "0.01",
      "--seed", "37",
      "--debug", "true",
      "--delim", ",",
      "--input", "src/test/resources/iris.csv",
      "--output", s"target/iris-result-${System.currentTimeMillis()}"))
  }
}
