import scala.collection.mutable

class TDM {

  var records : List[TDMRecord] =  List[TDMRecord]()

  def addTermToRecord(term : String)  =
  {
    //Find a record for the term
    val record =   records.find( x => x.term == term)
    if (record.nonEmpty)
    {
      val termRecord =  record.get
    termRecord.frequencyInAllDocuments +=1
    }
    else
    {
      //No record yet exists for this term
      val newRecord  = new TDMRecord(term, 1)
      records  = newRecord :: records
    }
  }
}
class TDMRecord(val term : String, var frequencyInAllDocuments :  Int )
{
  def log10Frequency : Double  = Math.log10(frequencyInAllDocuments.toDouble)
}