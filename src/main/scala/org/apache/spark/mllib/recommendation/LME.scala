package org.apache.spark.mllib.recommendation

package LME

import java.io.{FileOutputStream, OutputStreamWriter}

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.{ListBuffer, ArrayBuffer}
import scala.io.Source
import scala.util.Random


/*
* Distributed implementation of Latten Markov Embedding, more information about the
* algorithm can be found in the paper "Playlist Prediction via Metric Embedding",
* available at [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.480.9749&rep=rep1&type=pdf]
*
* */


//transNum: [ID,(number of from, number of to)]

private[LME] case class Model(ID: String,
                              vector: Array[Double],
                              popularity: Double)

class LME (var numOfFeatures: Int,
           var maxIteration: Int,
           var maxError: Double,
           var lambda: Double,
           var tau: Double,
           var usePopular: Boolean) extends Serializable{

  def this()=this(10, 100, 1e-3, 0.0, 20.0, true)

  def setFeatures(numOfFeatures: Int): this.type ={
    this.numOfFeatures = numOfFeatures
    this
  }

  def setIteration (numOfIteration: Int): this.type={
    this.maxIteration = numOfIteration
    this
  }

  def setMaxError (error: Double): this.type ={
    this.maxError = error
    this
  }

  def setLambda (lambdaValue: Double): this.type={
    this.lambda = lambdaValue
    this
  }

  def setTau (tauValue: Double): this.type ={
    this.tau = tauValue
    this
  }

  def setPopularity (ifConsider: Boolean): this.type ={
    this.usePopular = ifConsider
    this
  }

  var isFromFile = false
  def setInitialize(filePath: Boolean): this.type ={
    this.isFromFile = filePath
    this
  }

  var gaussian = false
  def gaussianRandom(isGaussian: Boolean) ={
    this.gaussian = isGaussian
  }

  var numOfRecommend = 10000
  def setRecommendNum(num: Int)={
    this.numOfRecommend = num
    this
  }

  var step = 0.0
  def setStep(x: Double)={
    this.step = x
    this
  }

  var mutableStep = false
  def setMutable(i: Boolean)={
    this.mutableStep = i
    this
  }

  var numOfPartition = 50
  def setPartitions(num:Int)={
    this.numOfPartition = num
    this
  }

  var persist: String = null
  var saveSteps = 50
  def setPersist(path: String, steps: Int)={
    this.persist = path
    this.saveSteps = steps
    this
  }

  def randomInitialize(random: Random): Array[Double]={
    var randomArray = new Array[Double](numOfFeatures)
    if(gaussian){
      randomArray.map(x=>random.nextGaussian())
    }else{
      randomArray.map(x=>random.nextDouble())
    }
  }

  def randDouble(): Double={
    math.random
  }

  def dataClean(rawData: RDD[(String, String)], numOfRecommend: Int):RDD[(String,String)]={
    val ids = rawData.flatMap(x=>Array(x._1,x._2)).
      map(x=>(x,1)).
      groupByKey().
      map{case(x,y)=>(x,y.size)}.
      filter(_._2>10).
      sortBy(x=>x._2,false).
      map(_._1).
      collect()
    var list = new ListBuffer[String]()
    var i = 0
    for(s<- ids){
      i+=1
      if(i<numOfRecommend) list+=s
    }
    rawData.filter(x=>list.contains(x._1)&&list.contains(x._2))
  }

  def readInitialize(ids: RDD[String]): RDD[Model]={
    val sc = ids.sparkContext
    /*val initialData = sc.textFile(path).
      map(x=>x.split(":")).
      map(x=>(x(0),(x(1).split(",").map(_.toDouble),x(2).toDouble)))
    val data = sc.broadcast(initialData.collectAsMap())*/
    val source = Source.fromFile(persist,"UTF-8")
    var map = new mutable.HashMap[String,(Array[Double],Double)]()
    val lineIterator = source.getLines()
    for(line<-lineIterator){
      val kv = line.split(":")
      map+=((kv(0),(kv(1).split(",").map(_.toDouble),kv(2).toDouble)))
    }
    source.close()
    val data = sc.broadcast(map)

    val initial = ids.map(id=>(id, {
      if(data.value.contains(id)){
        data.value(id)
      }else{
        val rand = new Random()
        (randomInitialize(rand),if(usePopular) randDouble() else 0.0)
      }
    })).map(x=>Model(x._1,x._2._1,x._2._2))
    data.unpersist(blocking = true)
    initial
  }

  def arrayPlus(a1: Array[Double], a2: Array[Double]):Array[Double]={
    val length = a1.length
    var a = new Array[Double](length)
    for(i<- 0 until length){
      a(i)=a1(i)+a2(i)
    }
    a
  }

  def arrayMinus(a1: Array[Double], a2: Array[Double]):Array[Double]={
    val length = a1.length
    var a = new Array[Double](length)
    for(i<- 0 until length){
      a(i)=a1(i)-a2(i)
    }
    a
  }

  def delta(X1: Array[Double], X2: Array[Double]): (Array[Double], Double)={
    val vector = arrayMinus(X1, X2)
    val mode = vector.map(x=>x*x).sum
    (vector, mode)
  }

  def saveResult(iterator: RDD[Model],path: String)={
    val save = iterator.map(model=>{
      var s = model.ID.toString
      s = s+":"
      for(i<- 0 until model.vector.length-1){
        s= s+model.vector(i)+","
      }
      s= s+model.vector(model.vector.length-1)+":"+model.popularity
      s
    }).collect
    val osw = new OutputStreamWriter(new FileOutputStream(path),"UTF-8")
    save.foreach(x=>osw.write(x+"\n"))
    osw.close()
  }

  def calcGlobalSum(iterator: RDD[Model],
                    globalFeature: Broadcast[collection.Map[String,(Array[Double],Double)]],
                    globalTransTo: Broadcast[collection.Map[String, Array[(String, Int)]]],
                    numOfTrans: Int):
  RDD[(String,Array[Double],Double,Double)]={

    iterator.map(model=>{
      var sum = 0.0
      var sumVector = new Array[Double](numOfFeatures)
      //      val localVector = globalFeature.value(model.ID)._1
      for(j<-globalFeature.value){
        val (vector, mode) = delta(model.vector, j._2._1)
        val tmp = math.exp(-mode+j._2._2)
        sum+= tmp
        sumVector = arrayPlus(sumVector,vector.map(x=>x*tmp))
      }
      if(globalTransTo.value.contains(model.ID)){
        val num = globalTransTo.value(model.ID).
          map(x=>x._2).
          sum.
          toDouble
        sumVector=sumVector.map(x=>x*2.0*num/sum)
      }else{
        sumVector=sumVector.map(x=>0.0)
      }
      (model.ID, arrayPlus(model.vector,sumVector.map(x=>x*step)),model.popularity,sum)
    })

  }


  def vectorDerivative(model: Model,
                       globalFeature: Broadcast[collection.Map[String,(Array[Double],Double)]],
                       globalSum: Broadcast[collection.Map[String,Double]],
                       globalTransFrom: Broadcast[collection.Map[String, Array[(String, Int)]]],
                       globalTransTo: Broadcast[collection.Map[String, Array[(String, Int)]]]): Array[Double]={

    val (localVector,localPopularity) = globalFeature.value(model.ID)

    var sum = new Array[Double](numOfFeatures)

    if(globalTransFrom.value.contains(model.ID)){
      for(a<- globalTransFrom.value(model.ID)){
        val (aVector, aPopularity) = globalFeature.value(a._1)
        val (deltaVector, factor) = delta(aVector, localVector)

        sum = arrayPlus(sum, deltaVector.map(x=>x*a._2*2.0))
      }
    }

    for(a<-globalFeature.value.toArray){
      val (deltaVector, factor) = delta(a._2._1, localVector)
      if(globalTransTo.value.contains(a._1)){
        val vector = deltaVector.map(x=>x*2.0*
          globalTransTo.value(a._1).map(x=>x._2).sum.toDouble*
          math.exp(-factor+localPopularity)/globalSum.value(a._1))
        sum = arrayMinus(sum, vector)
      }
    }

    if(globalTransTo.value.contains(model.ID)){
      for(a<- globalTransTo.value(model.ID)){
        sum = arrayPlus(sum, arrayMinus(globalFeature.value(a._1)._1,localVector).map(x=>x*a._2*2.0))
      }
    }

    sum = arrayMinus(sum,localVector.map(x=>x*2*lambda))
    sum
  }



  def popularityDerivative(model: Model,
                           globalFeature: Broadcast[collection.Map[String,(Array[Double],Double)]],
                           globalSum: Broadcast[collection.Map[String,Double]],
                           globalTransFrom: Broadcast[collection.Map[String, Array[(String, Int)]]],
                           globalTransTo: Broadcast[collection.Map[String, Array[(String, Int)]]]):Double={
    val (localVector,localPopularity) = globalFeature.value(model.ID)
    var sum = 0.0

    if(globalTransFrom.value.contains(model.ID)){
      sum+=globalTransFrom.value(model.ID).
        map(x=>x._2).
        sum.
        toDouble
    }

    for(a<-globalFeature.value.toArray){
      if(globalTransTo.value.contains(a._1)){
        sum-=globalTransTo.value(a._1).map(x=>x._2).sum.toDouble*
          math.exp(-delta(localVector,a._2._1)._2+localPopularity)/
          globalSum.value(a._1)
      }
    }


    //    Regulation for control of overfitting
    //    sum-= 2*lambda*localPopularity
    sum
  }

  def calcLikelihood(globalTransFrom: Broadcast[collection.Map[String, Array[(String, Int)]]],
                     globalTransTo: Broadcast[collection.Map[String, Array[(String, Int)]]],
                     globalFeature: Broadcast[collection.Map[String,(Array[Double],Double)]],
                     globalSum: Broadcast[collection.Map[String,Double]]): Double={
    var likelihood = 0.0

    for(a<- globalTransTo.value){
      for(b<- a._2){
        val aVector = globalFeature.value(a._1)._1
        val (bVector,bPopularity) = globalFeature.value(b._1)
        likelihood+=b._2.toDouble*(-delta(bVector,aVector)._2+bPopularity)
      }
      val num = a._2.map(x=>x._2).sum.toDouble
      likelihood-=num*math.log(globalSum.value(a._1))
    }
    likelihood
  }

  def run(rawData: RDD[(String, String)]): LMEResult = {
    val sc = rawData.sparkContext

    val data = if(numOfRecommend<=0) rawData else dataClean(rawData, numOfRecommend).
      cache()

    val numOfTrans = data.count().toInt

    val ids = data.flatMap(x=>Array(x._1,x._2)).distinct()


    var iterator = if(numOfPartition>0){
      if(isFromFile){
        readInitialize(ids)
      }else{
        if(usePopular){
          ids.mapPartitions(x=>{
            val rand = new Random()
            x.map(xx=>Model(xx,randomInitialize(rand),randDouble()))
          })
        }else{
          ids.mapPartitions(x=>{
            val rand = new Random()
            x.map(xx=>Model(xx,randomInitialize(rand),0.0))
          })
        }
      }.repartition(numOfPartition).persist()
    }else{
      if(isFromFile){
        readInitialize(ids)
      }else{
        if(usePopular){
          ids.mapPartitions(x=>{
            val rand = new Random()
            x.map(xx=>Model(xx,randomInitialize(rand),randDouble()))
          })
        }else{
          ids.mapPartitions(x=>{
            val rand = new Random()
            x.map(xx=>Model(xx,randomInitialize(rand),0.0))
          })
        }
      }.persist()
    }


    val transFrom = data.map{case(x,y)=>(y,x)}.
      groupByKey().
      map{case(next,previous)=>(next,{
      var map = new mutable.HashMap[String, Int]
      for(s<- previous){
        if(map.contains(s)){
          map(s)=map(s)+1
        }else {
          map.put(s,1)
        }
      }
      map.toArray
    })}

    val transTo = data.groupByKey().
      map{case(previous,next)=>(previous,{
      var map = new mutable.HashMap[String, Int]
      for(s<- next){
        if(map.contains(s)){
          map(s)=map(s)+1
        }else {
          map.put(s,1)
        }
      }
      map.toArray
    })}

    val globalTransFrom = sc.broadcast(transFrom.collectAsMap())
    val globalTransTo = sc.broadcast(transTo.collectAsMap())
    data.unpersist()

    var globalFeature = sc.broadcast(iterator.
      map(model=>(model.ID,(model.vector,model.popularity))).
      collectAsMap())

    var i = 0
    var error = 1.0
    var likelihood = -1.0

    while(i<maxIteration&&error>maxError){

      i+=1

      val calc = calcGlobalSum(iterator, globalFeature, globalTransTo, numOfTrans).persist()
      calc.setName(s"calc-$i")

      val globalSum = sc.broadcast(calc.map(x=>(x._1,x._4)).collectAsMap())

      /*val globalSum = sc.broadcast(
        calcGlobalSum(iterator, globalFeature, globalTransTo, numOfTrans, sc).
          collectAsMap())*/

      if(i<5){
        val tmpLikelihood = calcLikelihood(globalTransFrom, globalTransTo, globalFeature, globalSum)
        if(tmpLikelihood<likelihood&&step>1e-7){
          step=step/2.0
        }
        error = math.abs(tmpLikelihood-likelihood)
        likelihood = tmpLikelihood
        println(i+"\tlikelihood = "+ likelihood)
      }else if(i%5==0){
        val tmpLikelihood = calcLikelihood(globalTransFrom, globalTransTo, globalFeature, globalSum)
        if(tmpLikelihood<likelihood&&step>1e-7){
          step=step/2.0
        }
        error = math.abs(tmpLikelihood-likelihood)
        likelihood = tmpLikelihood
        println(i+"\tlikelihood = "+ likelihood)
      }

      if(mutableStep&&i%50==0){
        step=step/2.0
      }

      if(i%saveSteps==0&&persist!=null){
        /*        val calendar = Calendar.getInstance()
                val format = new SimpleDateFormat("HHmmss")
                val time = format.format(calendar.getTime)*/
        saveResult(iterator,persist)
      }

      iterator.unpersist()
      iterator = if(usePopular){
        calc.map(x=>Model(x._1,x._2,x._3)).map(model=>{
          val vectorChange = vectorDerivative(model,globalFeature,globalSum, globalTransFrom,globalTransTo)
          val popularityChange = popularityDerivative(model,globalFeature,globalSum, globalTransFrom,globalTransTo)
          Model(model.ID,
            arrayPlus(model.vector, vectorChange.map(x=>x*step)),
            model.popularity+popularityChange*step)
        })
      }else{
        calc.map(x=>Model(x._1,x._2,x._3)).map(model=>Model(model.ID,
          arrayPlus(model.vector,
            vectorDerivative(model,globalFeature,globalSum, globalTransFrom,globalTransTo).
              map(x=>x*step)),
          0.0)
        )
      }
      iterator.persist.setName(s"iter-$i")

      if((!sc.getCheckpointDir.isEmpty) && (i%10==0)) {
        iterator.checkpoint()
      }

      /*val previousIter = iterator

      iterator = if(usePopular){
        previousIter.map(model=>{
          val vectorChange = vectorDerivative(model,globalFeature,globalSum, globalTransFrom,globalTransTo)
          val popularityChange = popularityDerivative(model,globalFeature,globalSum, globalTransFrom,globalTransTo)
          Model(model.ID,
            arrayPlus(model.vector, vectorChange.map(x=>x*step)),
            model.popularity+popularityChange*step)
        })
      }else{
        previousIter.map(model=>Model(model.ID,
          arrayPlus(model.vector,
            vectorDerivative(model,globalFeature,globalSum, globalTransFrom,globalTransTo).
              map(x=>x*step)),
          0)
        )
      }*/

      val previousGlobalFeature = globalFeature
      val mapFeature = iterator.
        map(model=>(model.ID,(model.vector,model.popularity))).
        collectAsMap()
      globalFeature = sc.broadcast(mapFeature)
      previousGlobalFeature.unpersist(blocking = true)

      calc.unpersist()

      //      iterator.take(100).map(model=>println(model.ID+":"+model.vector(0)+":"+model.vector(1)+":"+model.popularity))

      globalSum.unpersist(blocking = true)

    }
    globalTransFrom.unpersist(blocking = true)
    globalTransTo.unpersist(blocking = true)
    new LMEResult(iterator.map(model=>(model.ID,(model.vector,model.popularity))))
  }
}

class LMEResult(resultData: RDD[(String, (Array[Double], Double))])
  extends Serializable{

  def arrayMinus(a1: Array[Double], a2: Array[Double]): Array[Double]={
    val length = a1.length
    var a = new Array[Double](length)
    for(i<- 0 until length){
      a(i)=a1(i)-a2(i)
    }
    a
  }

  def delta(X1: Array[Double], X2: Array[Double]): (Array[Double], Double)={
    val vector = arrayMinus(X1, X2)
    var sum = vector.map(x=>math.pow(x,2)).sum
    (vector, sum)
  }

  def feature(num: Int): RDD[(String, Array[(String, Double)])]={
    val sc = resultData.sparkContext
    val result = sc.broadcast(resultData.collectAsMap())

    val featureMatrix = resultData.map{case(id, feature)=>(id,{
      var arrayBuffer = new ArrayBuffer[(String, Double)]
      for(s<-result.value){
        if(!s._1.equals(id)){
          arrayBuffer+=((s._1, math.exp(-delta(s._2._1,feature._1)._2+feature._2)))
        }

      }
      val array = arrayBuffer.toArray
      val norm = array.map(x=>x._2).sum
      array.map(x=>(x._1,x._2/norm)).sortBy(x=>10.0-x._2).take(num)
    })}
    result.unpersist()
    featureMatrix
  }


  def mostPopular(num: Int): Array[String]={
    resultData.map(x=>(x._1,x._2._2)).sortBy(_._2,false).take(num).map(_._1)
  }

  def featureMatrix(num: Int): collection.Map[String, Array[(String,Double)]]={
    val result = resultData.collectAsMap()
    result.map(x=>(x._1,{
      var arrayBuffer = new ArrayBuffer[(String, Double)]
      for(s<-result){
        arrayBuffer.+=((s._1, math.exp(-delta(s._2._1,x._2._1)._2+x._2._2)))
      }
      val array = arrayBuffer.toArray
      val normalize = array.map(_._2).sum
      array.map(x=>(x._1,x._2/normalize)).sortBy(x=>10.0-x._2).take(num)
    }))
  }
}
