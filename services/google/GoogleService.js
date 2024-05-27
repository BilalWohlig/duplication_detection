const axios = require("axios");
const { Pinecone } = require("@pinecone-database/pinecone");

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_GOOGLE_API_KEY,
});
const pinecone_google = new Pinecone({
  apiKey: process.env.PINECONE_V2_GOOGLE_API_KEY,
});
const OpenAI = require("openai");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});
const { v4: uuidv4 } = require("uuid");
const index = pinecone.index("google-data");
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
    let stringData = Object.values(data);
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
    return await DummyData.find({similarityScore: {$gte: threshold}}).skip(skip).limit(limit)
  }

  async totalDocumentsInDB(page) {
    let pageNo = 1
    if(page) {
      pageNo = page
    }
    const limit = 20
    const skip = (pageNo - 1) * limit
    return await DummyData.find().skip(skip).limit(limit)
  }

  async totalDuplicateDocumentsInDB(page) {
    let pageNo = 1
    if(page) {
      pageNo = page
    }
    const limit = 20
    const skip = (pageNo - 1) * limit
    return await DummyData.find({status: 'duplicate'}).skip(skip).limit(limit)
  }

  async changeStatus(status, docId) {
    return await DummyData.findByIdAndUpdate(docId, {
      status: status
    }, {new: true})
  }

  async getOneRecord(recordId) {
    const record = await DummyData.findOne({_id: recordId, status: 'duplicate'}).populate("mostSimilarDocument")
    return {
      document: record.userData,
      duplicate: record.mostSimilarDocument.userData
    }
  }
}

module.exports = new GoogleService();
