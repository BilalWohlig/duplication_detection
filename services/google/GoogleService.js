const axios = require("axios");
const { Pinecone } = require("@pinecone-database/pinecone");

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_GOOGLE_API_KEY,
});
const pinecone_google = new Pinecone({
  apiKey: process.env.PINECONE_V2_GOOGLE_API_KEY,
});
const { Configuration, OpenAIApi } = require("openai");
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);
const { v4: uuidv4 } = require("uuid");
const index = pinecone.index("google-data");
const aiplatform = require("@google-cloud/aiplatform");
const { PredictionServiceClient } = aiplatform.v1;
const { helpers } = aiplatform; // helps construct protobuf.Value objects.

class GoogleService {
  async pushDataToPinecone(data, flag) {
    if (flag) {
      const batch_size = 100;
      for (let i = 0; i < data.length; i += batch_size) {
        const i_end = Math.min(data.length, i + batch_size);
        const meta_batch = data.slice(i, i_end);
        const ids_batch = meta_batch.map((x) => {
          return uuidv4();
        });
        const texts_batch = meta_batch.map((x) => {
          let stringData = Object.values(x);
          stringData = stringData.join(", ");
          return stringData;
        });
        let response;
        try {
          response = await openai.createEmbedding({
            model: "text-embedding-3-large",
            input: texts_batch,
          });
        } catch (error) {
          console.log("Error while creating embedding", error);
        }
        const embeds = response.data.data.map((record) => record.embedding);
        const meta_batch_cleaned = meta_batch.map((x) => ({
          FAMILYIDNO: x.FAMILYIDNO,
          AADHAR_REF_IDL: x.AADHAR_REF_ID,
          HOF_NAME_ENG: x.HOF_NAME_ENG,
          FATHER_NAME_ENG: x.FATHER_NAME_ENG,
          MOTHER_NAME_ENG: x.MOTHER_NAME_ENG,
          DOB: x.DOB,
          ALL_DATA: Object.values(x).join(", "),
        }));
        const to_upsert = ids_batch.map((id, i) => ({
          id: id,
          values: embeds[i],
          metadata: meta_batch_cleaned[i],
        }));
        await index.upsert(to_upsert);
        console.log("Successfully uploaded");
      }
    } else {
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
      const batch_size = 100;
      for (let i = 0; i < data.length; i += batch_size) {
        const i_end = Math.min(data.length, i + batch_size);
        const meta_batch = data.slice(i, i_end);
        const ids_batch = meta_batch.map((x) => {
          return uuidv4();
        });
        const texts_batch = meta_batch.map((x) => {
          let stringData = Object.values(x);
          stringData = stringData.join(", ");
          return stringData;
        });
        let embeds = [];
        try {
          const instances = texts_batch.map((e) =>
            helpers.toValue({ content: e, taskType: task })
          );
          const request = { endpoint, instances, parameters };
          const client = new PredictionServiceClient(clientOptions);
          const [response] = await client.predict(request);
          console.log("Got predict response");
          const predictions = response.predictions;
          // console.log(predictions);
          for (const prediction of predictions) {
            const embeddings = prediction.structValue.fields.embeddings;
            const values =
              embeddings.structValue.fields.values.listValue.values;
            const embeddingValues = values.map((value) => value.numberValue);
            embeds.push(embeddingValues);
          }
        } catch (error) {
          console.log("Error while creating embedding", error);
        }
        const meta_batch_cleaned = meta_batch.map((x) => ({
          FAMILYIDNO: x.FAMILYIDNO,
          AADHAR_REF_IDL: x.AADHAR_REF_ID,
          HOF_NAME_ENG: x.HOF_NAME_ENG,
          FATHER_NAME_ENG: x.FATHER_NAME_ENG,
          MOTHER_NAME_ENG: x.MOTHER_NAME_ENG,
          DOB: x.DOB,
          ALL_DATA: Object.values(x).join(", "),
        }));
        const to_upsert = ids_batch.map((id, i) => ({
          id: id,
          values: embeds[i],
          metadata: meta_batch_cleaned[i],
        }));
        await index.upsert(to_upsert);
        console.log("Successfully uploaded");
      }
    }
  }

  async getDataFromPinecone(data, flag) {
    let stringData = Object.values(data);
    stringData = stringData.join(", ");
    if (flag) {
      const embeddings = await openai.createEmbedding({
        model: "text-embedding-3-large",
        input: stringData,
      });
      var embeds = embeddings.data.data.map((record) => record.embedding);
      const finalResponse = index.query({
        vector: embeds[0],
        topK: 5,
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
        topK: 5,
        includeValues: false,
        includeMetadata: true,
      });
      return finalResponse;
    }
  }
}

module.exports = new GoogleService();
