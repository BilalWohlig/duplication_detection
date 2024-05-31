const axios = require("axios");
const { Pinecone } = require("@pinecone-database/pinecone");
const mongoose = require('mongoose')

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_GOOGLE_API_KEY,
});
const pinecone_google = new Pinecone({
  apiKey: process.env.PINECONE_V2_GOOGLE_API_KEY,
});
const pinecone_v3 = new Pinecone({
  apiKey: process.env.PINECONE_V3_GOOGLE_API_KEY
})
const OpenAI = require("openai");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});
const { v4: uuidv4 } = require("uuid");
const index = pinecone_v3.index("google-data");
const aiplatform = require("@google-cloud/aiplatform");
const { PredictionServiceClient } = aiplatform.v1;
const { helpers } = aiplatform; // helps construct protobuf.Value objects.
const DummyData = require("../../mongooseSchema/DummyData")
const { faker } = require("@faker-js/faker")
const randomizer = require("randomatic")
const fs = require("fs")
const { TranslationServiceClient } = require("@google-cloud/translate");
if (process.env.GOOGLE_TRANSLATE_CREDENTIALS) {
  try {
    var translateKey = JSON.parse(process.env.GOOGLE_TRANSLATE_CREDENTIALS);
  } catch (error) {
    console.error("Error reading the JSON file:", error);
  }
}

class GoogleService {

  async getDataFromPinecone(data, flag) {
    let finalObj = {
      "FAMILYIDNO": data.FAMILYIDNO,
      "ENROLLMENT_ID": data.ENROLLMENT_ID,
      "MEMBER_ID": data.MEMBER_ID,
      "AADHAR_REF_ID": data.AADHAR_REF_ID,
      "HOF_NAME_ENG": data.HOF_NAME_ENG,
      "FATHER_NAME_ENG": data.FATHER_NAME_ENG,
      "DOB": data.DOB,
      "MOTHER_NAME_ENG": data.MOTHER_NAME_ENG,
      "SPOUSE_NAME_ENG": data.SPOUSE_NAME_ENG,
      "MOBILE_NO": data.MOBILE_NO,
      "EMAIL": data.EMAIL,
      "HOF_ACCOUNT_NO": data.HOF_ACCOUNT_NO,
      "ADDRESS_CO_ENG": data.ADDRESS_CO_ENG,
      "MNAREGA_NO": data.MNAREGA_NO,
      "ELECTRICITY_CON_ID": data.ELECTRICITY_CON_ID,
      "WATER_BILL_NO": data.WATER_BILL_NO,
      "GAS_CON_NO": data.GAS_CON_NO,
      "RATION_CARD_NO": data.RATION_CARD_NO,
      "RATION_MEM_UID": data.RATION_MEM_UID,
      "BPL_CARD_NO": data.BPL_CARD_NO,
      "EMPLOYEMENT_REG_NO": data.EMPLOYEMENT_REG_NO,
      "GOVT_EMP_ID": data.GOVT_EMP_ID,
      "SSP_PPO_NO": data.SSP_PPO_NO,
      "LABOUR_CARD_NO": data.LABOUR_CARD_NO,
      "VOTER_ID_NO": data.VOTER_ID_NO,
      "DRIVING_LIC_NO": data.DRIVING_LIC_NO,
      "PASSPORT_ID": data.PASSPORT_ID,
      "PAN_CARD_NO": data.PAN_CARD_NO,
    }
    let stringData = Object.values(finalObj);
    stringData = stringData.join(", ");
    if (flag) {
      const embeddings = await openai.embeddings.create({
        model: "text-embedding-3-large",
        input: stringData,
      });
      var embeds = embeddings.data.map((record) => record.embedding);
      const finalResponse = index.query({
        vector: embeds[0],
        topK: 1,
        includeValues: false,
        includeMetadata: true,
      });
      return finalResponse;
    } else {
      let embeds = []
      const index = pinecone_google.index("google-data");
      const project = process.env.PROJECT_ID;
      const apiEndpoint = "us-central1-aiplatform.googleapis.com";
      const outputDimensionality = 0;
      const model = "text-embedding-004";
      const task = "SEMANTIC_SIMILARITY";
      const clientOptions = { apiEndpoint: apiEndpoint };
      const location = process.env.EMBEDDING_LOCATION;
      const endpoint = `projects/${project}/locations/${location}/publishers/google/models/${model}`;
      const parameters =
        outputDimensionality > 0
          ? helpers.toValue(outputDimensionality)
          : helpers.toValue(3072);
      // const instances = texts_batch.map((e) =>
      //   helpers.toValue({ content: e, taskType: task })
      // );
      const instances = [helpers.toValue({ content: stringData, taskType: task })]
      const request = { endpoint, instances, parameters };
      const client = new PredictionServiceClient(clientOptions);
      const [response] = await client.predict(request);
      console.log("Got predict response");
      const predictions = response.predictions;
      // console.log(predictions);
      for (const prediction of predictions) {
        const embeddings = prediction.structValue.fields.embeddings;
        const values = embeddings.structValue.fields.values.listValue.values;
        const embeddingValues = values.map((value) => value.numberValue);
        embeds.push(embeddingValues);
      }
      const finalResponse = index.query({
        vector: embeds[0],
        topK: 1,
        includeValues: false,
        includeMetadata: true,
      });
      return finalResponse;
    }
  }

  async getRecordsAboveThreshold(threshold, page) {
    let pageNo = 1
    if(page) {
      pageNo = page
    }
    const limit = 20
    const skip = (pageNo - 1) * limit
    const docs = await DummyData.find({similarityScore: {$gte: threshold}, status: {$exists: false}}).skip(skip).limit(limit)
    const totalDocs = await DummyData.find({similarityScore: {$gte: threshold}, status: {$exists: false}})
    for(const doc of docs) {
      doc.userData.similarityScore = doc.similarityScore
    }
    return {
      documents: docs,
      totalCount: totalDocs.length
    }
  }

  async totalDocumentsInDB() {
    const docs = await DummyData.find()
    return docs.length
  }

  async totalDuplicateDocumentsInDB(page) {
    let pageNo = 1
    if(page) {
      pageNo = page
    }
    const limit = 20
    const skip = (pageNo - 1) * limit
    const docs = await DummyData.find({status: 'duplicate'}).skip(skip).limit(limit)
    const totalDocs = await DummyData.find({status: 'duplicate'})
    return {
      documents: docs,
      totalCount: totalDocs.length
    }
  }

  async changeStatus(status, ids) {
    for(const id of ids) {
      await DummyData.findByIdAndUpdate(id, {
        status: status,
        userDefinedDuplicate: status == 'duplicate' ? true : false
      }, {new: true})
    }
    return "Doneee"
  }

  async getOneRecord(recordId) {
    const record = await DummyData.findOne({_id: recordId, status: {$exists: false}}).populate("mostSimilarDocument")
    if(record) {
      const keys = Object.keys(record.userData)
      let objArray = []
      for(const key of keys) {
        const obj = {
          columnKey: key,
          document: record.userData[key],
          duplicate: record.mostSimilarDocument.userData[key],
          isSame: false
        }
        if(obj.document == obj.duplicate) {
          obj.isSame = true
        }
        objArray.push(obj)
      }
      return {
        comparison: objArray,
        similarityScore: record.similarityScore
      }
    }
    return {
      comparison: [],
      similarityScore: 0
    }
  }
  
  async getOneRecordV2(recordId) {
    const record = await DummyData.findById(recordId);
    const docs = await DummyData.aggregate([
      {
        $match: {
          _id: mongoose.Types.ObjectId(record.mostSimilarDocument),
        }
      },
      {
        $lookup: {
          from: "dummydatas",
          let: { originalId: "$_id" },
          pipeline: [
            {
              $match: {
                similarityScore: { $gte: 90 }
              }
            },
            {
              $match: {
                $expr: {
                  $eq: ["$mostSimilarDocument", "$$originalId"]
                }
              }
            }
          ],
          as: "duplicates"
        }
      },
      {
        $addFields: {
          duplicates: {
            $filter: {
              input: "$duplicates",
              as: "doc",
              cond: { $ne: ["$$doc", null] }
            }
          }
        }
      },
    ])
    let finalArray = []
    docs[0].userData._id = docs[0]._id
    docs[0].userData.recordTitle = 'Original'
    docs[0].userData.similarityScore = 0
    finalArray.push(docs[0].userData)
    for (let i = 0; i < docs[0].duplicates.length; i++) {
      const duplicate = docs[0].duplicates[i];
      duplicate.userData._id = duplicate._id
      duplicate.userData.recordTitle = `Duplicate ${i+1}`
      duplicate.userData.similarityScore = duplicate.similarityScore
      finalArray.push(duplicate.userData)
    }
    return finalArray;
  }
}

module.exports = new GoogleService();
