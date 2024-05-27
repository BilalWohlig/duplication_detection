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
          response = await openai.embeddings.create({
            model: "text-embedding-3-large",
            input: texts_batch,
          });
        } catch (error) {
          console.log("Error while creating embedding", error);
        }
        const embeds = response.data.map((record) => record.embedding);
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
        console.log("Successfully uploaded", i/100);
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
      const batch_size = 50;
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
          // console.log("Got predict response");
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
        console.log("Successfully uploaded", i/50);
      }
    }
  }

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

  async generateUniqueNumericId(length) {
    return randomizer('0', length).toString();
  }

  async generateUniqueAlphanumericId(length) {
    return randomizer('A0', length).toString();
  }

  async translateText(text, lang) {
    const translationClient = new TranslationServiceClient(
      translateKey
        ? {
            credentials: translateKey,
            projectId: translateKey.project_id,
          }
        : {}
    );
    const request = {
      parent: `projects/${process.env.PROJECT_ID}/locations/${process.env.PROJECT_LOCATION}`,
      contents: [text],
      mimeType: "text/plain", // mime types: text/plain, text/html
      sourceLanguageCode: "en",
      targetLanguageCode: lang,
    };

    // Run request
    const [response] = await translationClient.translateText(request);
    return response.translations[0].translatedText;
  }

  async generateDummyData(number) {
    let userArray = []
    let gender = "male"
    try {
      for (let i = 0; i < number; i++) {
        // const userData = await this.generateUsersFromGPT(number)
        // return JSON.parse(userData.choices[0].message.content)
        const finalObj = await this.generateUniqueUser(gender)
        userArray.push({userData: finalObj})
        console.log("User", i + 1)
        // userArray.push(await this.generateUniqueUser(gender))
        if(gender == 'male'){
          gender = 'female'
        }
        else{
          gender = 'male'
        }
      }
    } catch (error) {
      console.log(error)
    }
    await DummyData.insertMany(userArray)
    console.log("MongoDB Doneeeeee", userArray.length)
    const pineconeArray = userArray.map((user) => user.userData)
    // await this.pushDataToPinecone(pineconeArray, true)
    // console.log("Pineconeeee Doneeeeee", pineconeArray.length)
    return userArray
  }
  
  async generateUniqueUser(gender) {
    const firstName = faker.person.firstName({sex: gender})
    const lastName = faker.person.firstName({sex: gender})
    const fullName = firstName + " " + lastName
    const fatherName = faker.person.fullName({lastName: lastName, sex: 'male'})
    const motherName = faker.person.fullName({lastName: lastName, sex: 'female'})
    const address = faker.location.streetAddress(true)
    const city = faker.location.city()
    const cityBlock = randomizer('A', 1)
    const caste = faker.helpers.arrayElement(["Brahmin", "Kshatriya", "Vaishya", "Shudras"])
    const casteHindi = await this.translateText(caste, "hi")
    let totalFamilyMember

    const maritalStatus = faker.helpers.arrayElement(["Single", "Married", "Divorced", "Widowed"])
    let spouseName
    let spouseNameHindi
    let relationType
    if(maritalStatus == 'Single' || maritalStatus == 'Divorced'){
      spouseName = '',
      spouseNameHindi = ''
      if(maritalStatus == 'Single') {
        totalFamilyMember = "1"
      }
      else{
        totalFamilyMember = faker.number.int({ min: 1, max: 10 }).toString()
      }
    }
    else{
      if(gender == 'male'){
        spouseName = faker.person.fullName({lastName: lastName, sex: 'female'})
      }
      else{
        spouseName = faker.person.fullName({lastName: lastName, sex: 'male'})
      }
      spouseNameHindi = await this.translateText(spouseName, 'hi')
      if(maritalStatus == 'Married') {
        totalFamilyMember = faker.number.int({ min: 2, max: 10 }).toString()
      }
      else{
        totalFamilyMember = faker.number.int({ min: 1, max: 10 }).toString()
      }
    }
    if(gender == 'male') {
      relationType = 'Son'
    }
    else{
      relationType = 'Daughter'
    }
    return {
      "FAMILYIDNO": await this.generateUniqueNumericId(10),
      "ENROLLMENT_ID": await this.generateUniqueAlphanumericId(15),
      "MEMBER_ID": await this.generateUniqueNumericId(11),
      "AADHAR_REF_ID": await this.generateUniqueNumericId(15),
      "HOF_NAME_ENG": firstName + " " + lastName,
      "HOF_NAME_HND": await this.translateText(fullName, "hi"),
      "FATHER_NAME_ENG": fatherName,
      "FATHER_NAME_HND": await this.translateText(fatherName, "hi"),
      "DOB": faker.date.past({years: 50}, {refDate: new Date()}).toISOString().slice(0, 10).split('-').reverse().join('-'),
      "MOTHER_NAME_ENG": motherName,
      "MOTHER_NAME_HND": await this.translateText(motherName, "hi"),
      "GENDER": gender,
      "MARITAL_STATUS": maritalStatus,
      "SPOUSE_NAME_ENG": spouseName,
      "SPOUSE_NAME_HND": spouseNameHindi,
      "MOBILE_NO": faker.phone.number().toString(),
      "EMAIL": faker.internet.email({firstName: firstName, lastName: lastName}),
      "DISABILITY_CAT": faker.helpers.arrayElement(["None", "Physical", "Visual", "Hearing", "Intellectual"]),
      "HOUSE_CATEGORY": faker.helpers.arrayElement(["Owned", "Rented"]),
      "CATEGORY": faker.helpers.arrayElement(["General", "OBC", "SC", "ST"]),
      "CASTE_HND": casteHindi,
      "IS_MINORITY": faker.helpers.arrayElement(["Yes", "No"]),
      "RELIGION": faker.helpers.arrayElement(["Hindu", "Muslim", "Christian", "Sikh", "Buddhist", "Jain", "Jew"]),
      "EDUCATION": faker.helpers.arrayElement(["Illiterate", "Primary", "Secondary", "Graduate", "Postgraduate"]),
      "RESIDENTIAL_CODE":faker.location.zipCode(),
      "LIVING_SINCE_YEAR": faker.number.int({ min: 1900, max: 2024 }).toString(),
      "RELATION_TYPE": relationType,
      "TOTAL_FAMILY_MEMBER": totalFamilyMember,
      "HOF_ACCOUNT_NO": await this.generateUniqueNumericId(15),
      "HOF_BRANCH_NAME": faker.helpers.arrayElement(["HDFC", "SBI", "ICICI", "BOB", "DCB", "YES", "HSBC"]),
      "ADDRESS_CO_ENG": address,
      "ADDRESS_CO_HND": await this.translateText(address, "hi"),
      "STATE": faker.location.state(),
      "DISTRICT": city,
      "IS_RURAL": faker.helpers.arrayElement(["Yes", "No"]),
      "CITY_BLOCK": cityBlock,
      "GRAM_PANCHAYAT": "",
      "WARD_HND": await this.translateText(`Ward ${cityBlock}`, "hi"),
      "VILLAGE_ENG": faker.location.city(),
      "PIN_CODE": faker.location.zipCode().toString(),
      "MNAREGA_NO": await this.generateUniqueNumericId(12),
      "ELECTRICITY_CON_ID": await this.generateUniqueNumericId(10),
      "WATER_BILL_NO": await this.generateUniqueNumericId(10),
      "GAS_AGENCY": faker.helpers.arrayElement(["Bharat Gas", "HP Gas", "Jai Balaji", "Kamal Gas", "Amber Gas"]),
      "GAS_CON_NO": await this.generateUniqueNumericId(10),
      "RATION_CARD_NO": await this.generateUniqueNumericId(10),
      "RATION_MEM_UID": await this.generateUniqueNumericId(10),
      "RATION_CARD_TYPE": faker.helpers.arrayElement(["APL", "BPL", "AAY"]),
      "BPL_CARD_NO": await this.generateUniqueNumericId(10),
      "HOUSE_TYPE": faker.helpers.arrayElement(["Kutcha", "Pucca", "Semi-Pucca"]),
      "HOUSE_STATUS": faker.helpers.arrayElement(["Owned", "Rented"]),
      "EMPLOYEMENT_REG_NO": await this.generateUniqueNumericId(10),
      "GOVT_EMP_ID": await this.generateUniqueNumericId(10),
      "SSP_PPO_NO": await this.generateUniqueNumericId(10),
      "YEARLY_EXACT_INCOME": faker.number.int({ min: 10000, max: 1000000 }).toString(),
      "OCCUPATION": faker.helpers.arrayElement(["Farmer", "Laborer", "Service", "Business", "Unemployed"]),
      "OCCUPATION_SUB_CAT": faker.person.jobTitle(),
      "LABOUR_CARD_NO": await this.generateUniqueNumericId(10),
      "LABOUR_CARD_END_DATE": faker.date.future().toISOString().slice(0, 10).split('-').reverse().join('-'),
      "IS_ORPHAN": faker.helpers.arrayElement(["Yes", "No"]),
      "VOTER_ID_NO": await this.generateUniqueAlphanumericId(10),
      "DRIVING_LIC_NO": await this.generateUniqueAlphanumericId(10),
      "PASSPORT_ID": await this.generateUniqueAlphanumericId(8),
      "PAN_CARD_NO": await this.generateUniqueAlphanumericId(10),
      "NFSA": faker.helpers.arrayElement(["Yes", "No"])
    }
  }

  async getRecordsAboveThreshold(threshold) {
    const records = await DummyData.find()
  }

  async updateAllDocuemntsInDB() {
    const records = await DummyData.find({pineconeId: {$exists: true}})
    for (let i = 0; i < records.length; i++) {
      const record = records[i];
      const document = await index.query({
        vector: [0.09,0.49,0.68,0.53,0.05,1,0.04,0.84,0.07,0.93,0.61,0.12,0.78,0.84,0.9,0.46,0.83,0.71,0.76,0.2,0.51,0.38,0.46,0.07,0.74,0.19,0.26,0.23,0.76,0.66,0.76,0.05,0.62,0.69,0.88,0.05,0.38,0.25,0.92,0.04,0.54,0.98,0.4,0.92,0.39,0.73,0.51,0.05,0.61,1,0.21,0.46,0.16,0.25,0.31,0.69,0.44,0.32,0.99,0.4,0.77,0.79,0.72,0.97,0.06,0.65,0.72,0.12,0.5,0.53,0.57,0.76,0.94,0.35,0.92,0.8,0.07,0.42,0.5,0.23,0.96,0.78,0.49,0.48,0.84,0.27,0.54,0.57,0.95,0.6,0.75,0.43,0.52,0.19,0.72,0.95,0.38,0.28,0.02,0.75,0.09,0.44,0.82,0.61,0.61,0.04,0.8,0.26,0.08,0.5,0.35,0.48,0.79,0.53,0.59,0.74,0.48,0.81,0.92,0.64,0.1,0.24,0.38,0.04,0.23,0.45,0.56,0.89,0.36,0.96,0.05,0.92,0.59,0.21,0.61,0.45,0.86,0.35,0.96,0.42,0.6,0.06,0.15,0.96,0.17,0.78,0.92,0.22,0.54,0.2,0.14,0.85,0.95,0.14,0.95,0.26,0.16,0.78,0.64,0.94,0.24,0.08,0.4,0.28,0.34,0.97,0.51,0.12,0.95,0.43,0.12,0.74,0.39,0.32,0.66,0.26,0.48,0.14,0.42,0.94,0.38,0.84,0.88,0.75,0.77,0.36,0.74,0.83,0.84,0.04,0.08,0.44,0.13,0.48,0.72,0.63,0.43,0.06,0.62,0.76,0.22,0.88,0.34,0.71,0.08,0.01,0.61,0.44,0.65,0.12,0.5,0.21,0.74,0.52,0.18,0.66,0.51,0.36,0.61,0.03,0.43,0.26,0.78,0.28,0.4,0.26,0.21,0.43,0.2,0.2,0.87,0.47,0.05,0.74,0.84,0.58,0.87,0.13,0.65,0.73,0.93,0.38,0.82,0.94,0.71,0.52,0.22,0.19,0.99,0.52,0.85,0.52,0.29,0.29,0.09,0.99,0.59,0.25,0.3,0.04,0.04,0.3,0.4,0.39,0.29,0.93,0.01,0.8,0.45,0.38,0.65,0.03,0.85,0.92,0.83,0.17,0.77,0.64,0.12,0,0.87,0.2,0.22,0.78,0.78,0.08,0.8,0.35,0.45,0.29,0.37,0.69,0.92,0.08,0.3,0.49,0.34,0.43,0.36,0.91,0.47,0.07,0.29,0.75,0.99,0.56,0.12,0.04,0.27,0.92,0.19,0.67,0.27,0.37,0.34,0.3,0.04,0,0.72,0.3,0.78,0.33,0.64,0.51,0.14,0.66,0.45,0.93,0.77,0.68,0.69,0.6,0.83,0.42,0.96,0.74,0.64,0.23,0.28,0.88,0.93,0.42,0.74,0.24,0.21,0.69,0.17,0.69,0.22,0.13,0.79,0.52,0.72,0.41,0.01,0.12,0.87,0.25,0.3,0.99,0.87,0.82,0.84,0.03,0.07,0.44,0.99,1,0.59,0.16,0.78,0.9,0.98,0.73,0.37,0.02,0.56,0.53,0.24,0.58,0.52,0.75,0.09,0.36,0.72,0.6,0.41,0.11,0.01,0.64,0.26,0.81,0.47,0.15,0.49,0.59,0.44,0.59,0.97,0.1,0.37,0.84,0.31,0.19,0.76,0.88,0.04,0.31,0.23,0.55,0.23,0,0.8,0.95,0.61,0.89,0.12,0.84,0.49,0.05,0.74,0.36,0.96,0.12,0.69,0.53,0.14,0.12,0.18,0.29,0.82,0.66,0.47,0.51,0.74,0.33,0.87,0.08,0.09,0.74,0.18,0.04,0.78,0.53,0.86,0.18,0.43,0.88,0.29,0.95,0.53,0.88,0.42,0.31,0.38,0.69,0.31,0.68,0.99,0.65,0.66,0.54,0.29,0.84,0.35,0.53,0.18,0.14,0.68,0.43,0.76,0.1,0.61,0.33,0.07,0.3,0.27,0.1,0.04,0.04,0.75,0.89,0.09,0.18,0.04,0.87,0.64,0.33,0.66,0.65,0.17,0.4,0.13,0.14,0.14,0.54,0.63,0.38,0.63,0.09,0.28,0.53,0.38,0.2,0.46,0.7,0.73,0.27,0.88,0.76,0.38,0.42,0.95,0.31,0.8,0.83,0.34,0,0.38,0.22,0.83,0.65,0.51,0.14,0.62,0.33,0.76,0.7,0.75,0.57,0.76,0.72,0.69,0.33,0.35,0.9,0.05,0.84,0.15,0.08,0.86,0.27,0.7,0.74,0.16,0.54,0.39,0.04,0.26,0.28,0.45,0.19,0.92,0.28,0.54,0.43,0.8,0.7,0.4,0.31,0.31,0.52,0.85,0.54,0.29,0.21,0.26,0.29,0.91,0.14,0.9,0.55,0.9,0.72,0.58,0.86,0.72,0.52,0.68,0.77,0.73,0.2,0.1,0.85,0.71,0.54,0.66,0.97,0.67,0.69,0.12,0.3,0.74,0.81,0.46,0.48,0.77,0.99,0.28,0.49,0.97,0.57,0.93,0.98,0.4,0.31,0.91,0.01,0.25,0.21,0.18,0.96,0.39,0.78,0.86,0.11,0.22,0.21,0.72,0.27,0.36,0.92,0.65,0.28,0.47,0.1,0.95,0.26,0.37,0.31,0.65,0.14,0.97,0.81,0.43,0.92,0.04,0.13,0.34,0.89,0.13,0.78,0.91,0.35,0.12,0.9,0.28,0.37,0.82,0.94,0.68,0.93,0.32,0.3,0.77,0.88,0.51,0.5,0.09,0.44,0.56,0.86,0.74,0.22,0.22,0.89,0.2,0.13,0.04,0.11,0.24,0.99,0.64,0.4,0.09,0.86,0.77,0.24,0.19,0.6,0.28,0.54,0.13,0.61,0.67,0.81,0.2,0.12,0.9,0.26,0.77,0.7,0.77,0.49,0.04,0.11,0,0.46,0.61,0.69,0.31,0.6,0.65,0.72,0.01,0.66,0.22,0.82,0.29,0.52,0.55,0.24,0.08,0.42,0.61,0.58,0.33,0.9,0.4,0.52,0.44,0.52,0.18,0.39,0.39,0.9,0.42,0.58,0.74,0.31,0.81,0.93,0.14,0.16,0.14,0.52,0.77,0.79,0.2,0.28,0.95,0.69,0.39,0.55,0.69,0.54,0.03,0.65,0.68,0.61,0.26,1,0.76,0.28,0.13,0.91,0.07,0.35,0.66,0.55,0.45,0.58,0.59,0.85,0.2,0.11,0.68,0.83,0.44,1,0.68,0.96,0.5,0.87,0.66,0.88,0.68,0.05,0.75,0.85,0.04,0.09,0.96,0.31,0.57,0.55,0.69,0,0.42,0.72,0.48,0.66,0.2,0.51,0.46,0.43,0.6,0.17,0.13,0.44,0.03,0.34,0.09,0.04,0.66,0.42,0.65,0.57,0.13,0.53,0.81,0.73,0.65,0.79,0.9,0.63,0.61,0.7,0.81,0.39,0.13,0.79,0.26,0.53,0.43,0.26,0.89,0.54,0.88,0.24,0.41,0.1,0.27,0.42,0.22,0.64,0.31,0.8,0.25,0.58,0.83,0.92,0.35,0.02,0.24,0,0.7,0.31,0.6,0.33,0.68,0.55,0.43,0.19,0.47,0.43,0.11,0.26,0.81,0.61,0.93,0.27,0.92,0.06,0.09,0.59,0.03,0.64,0.1,0.42,0.45,0.63,0.5,0.91,0.63,0.56,0.71,0.24,0.62,0.74,0.31,0.85,0.64,0.75,0.96,0.7,0.7,0.94,0.88,0.65,0.99,0.1,0.79,0.74,0.18,0.49,0.57,0.8,0.73,0.83,0.16,0.8,0.26,0.3,0.45,0.56,0.94,0.73,0.23,0.45,0.14,0.58,0.5,0.29,0.82,0.71,0.42,0.54,0.36,0.9,0.77,0.25,0.52,0.41,0.4,0.29,0.29,0.83,0.65,0.17,0.95,0.67,0.97,0.33,0.03,0.48,0.25,0.3,0.24,0.85,0.67,0.17,0.62,0.59,0.1,0.99,0.19,0.16,0.21,0.22,0.13,0.05,0.79,0.16,0.91,0.24,0.28,0.93,0.85,0.26,0.84,0.03,0.6,0.49,0.79,0.1,0.91,0.58,0.8,0.38,0.33,0.38,0.91,0.32,0.97,0.57,0.11,0.26,0,0.13,0.75,0.33,0.85,0.85,0.78,0.87,0.01,0.26,0.06,0.96,0.02,0.61,0.12,0.94,0.08,0.23,0.21,0.8,0.19,0.79,0.85,0.25,0.3,0.76,0.41,0.25,0.18,0.61,0.01,0.63,0.89,0.68,0.72,0.56,0.34,0.69,0.02,0.82,0.42,0.25,0.17,0.94,0.58,0.8,0.38,0.23,0.77,0.86,0.63,0.88,0.44,0.54,1,0.71,0.14,0.75,0.01,0.79,0.4,0.17,0.03,0.61,0.46,0.48,0.96,0.51,0.37,0.98,0.94,0.66,0.38,0.52,0.98,0.86,0.04,0.92,0.51,0.58,0.97,0.64,0.32,0.75,0.28,0.69,0.98,0.37,0.85,0.34,0.19,0.65,0.36,0.03,0.13,0.01,0.85,0.72,0.31,0.41,0.33,0.84,0.62,0.65,0.57,0.68,0.52,0.32,0.6,0.91,0.76,0.97,0.44,0,0.61,0.81,0.92,0.37,0.73,0.64,0.55,0.05,0.14,0.92,0.65,0.87,0.83,0.7,0.12,0.61,0.11,0.08,0.25,0.38,0.05,0.22,0.16,0.02,0.32,0.29,0.89,0.36,0.92,0.22,0.69,0.23,0.36,0.49,0.94,0.23,0.98,0.7,0.99,0.68,0.35,0.32,0.07,0.9,0.57,0.12,0.05,0.32,0.05,0.18,0.67,0.04,0.61,0.59,0.56,0.52,0.56,0.24,0.09,0.58,0.24,0.23,0.48,0.98,0.33,0.08,0.52,0.43,0.07,0.87,0.58,0.14,0.4,0.31,0.57,0.4,0.36,0.78,0.07,0.43,0.73,0.12,0.45,0.1,0.69,0.25,0.65,0.96,0.28,0.28,0.36,0.6,0.27,0.66,0.79,0.46,0.24,0.97,0.49,0.67,0.11,0.48,0.7,0.88,0.56,0.41,0.76,0.61,0.85,0.19,0.29,0.7,0.16,0.13,0.25,0.63,0.65,0.45,0.29,0.78,0.75,0.19,0.9,0.76,0.56,0.85,0.5,0.65,0.36,0.51,0.74,0.51,0.25,0.13,0.52,0.53,0.86,0.27,0.09,0.27,0.79,0.94,0.9,1,0.31,0.66,0.73,0.02,0.45,0.17,0.09,0.54,0.03,0.23,0.09,0.28,0.17,0.71,0.69,0.57,0.81,0.57,0.15,0.81,0.07,0.86,1,0.55,0.43,0.83,0.57,0.99,0.52,0.14,0.4,0.49,0.5,0.93,0.32,0.31,0.83,0.15,0.58,0.32,0.63,0.64,0.46,0.96,0.22,0.17,0.4,0.54,0.85,0.74,0.17,0.49,0.35,0.07,0.9,0.57,0.74,0.77,0.18,0.39,0.13,0.71,0.82,0.63,0.64,0.97,0.94,0.12,0.6,0.95,0.11,0.36,0.28,0.72,0.52,0.24,0.22,0.71,0.57,0.17,0.98,0.55,0.82,0.02,0.87,0.22,0.64,0.61,0.72,0.38,0.49,0.8,0.79,0.57,0.39,0.45,0.51,0.49,0.54,0.26,0.19,0.35,0.17,0.23,0.24,0.88,0.11,0.37,0.69,0.54,0.87,0.31,0,0.4,0.92,0.18,0.28,0.24,0.99,0.58,0.84,0.04,0.52,0.24,0.4,0.45,0.4,0.56,0.36,0.37,0.9,0.63,0.06,0.36,0.98,0.69,0.16,0.01,0.97,0.13,0.62,0.44,0.8,0.24,0.1,0.93,0.52,0.24,0.29,0,0.05,0.02,0.98,0.14,0.17,0.79,0.04,0.63,0.14,0.65,0.23,0.74,0.03,0.07,0.1,0.85,0.76,0.3,0.54,0.32,0.42,0.39,0.4,0.58,0.02,0.43,0.59,0.92,0.66,0.54,1,0.33,0.43,0.1,0.24,0.03,0.62,0.64,0.2,0.82,0.89,0.53,0.26,0.1,0.36,0.88,0.14,0.36,0.2,0.85,0.84,0.8,0.28,0.89,0.94,0.69,0.68,0.28,0.22,0.54,0.46,0.97,0.72,0.72,0.38,0.2,0.65,0.71,0.92,0.4,0.36,0.9,0.41,0.05,0.8,0.41,0.44,0.77,0.64,0.49,0.61,0.4,0.22,0.19,0.06,0.7,0.32,0.77,0.01,0.21,0.41,0.56,0.2,0.56,0.31,0.36,0.59,0.07,0.32,0.71,0.96,0.98,0.76,0.68,0.57,0.66,0.24,0.5,0.04,0.71,0.06,0.44,0.34,0.22,0.94,0.6,0.09,0.58,0.49,0.94,0.38,0.21,0.85,0.5,0.88,0.62,0.44,0.22,0.58,0.17,0.75,0.29,0.62,0.9,0.53,0.35,0.46,0.24,0.23,0.92,0.87,0.35,0.8,0.48,0.85,0.87,0.9,0.87,0.98,0.88,0.4,0.28,0.43,0.2,0.16,0.12,0.99,0.63,0.35,0.46,0.73,0.26,0.4,0.5,0.67,1,0.09,1,0.81,0.09,0.12,0.08,0.24,0.53,0.2,0.74,0.34,0.71,0.23,0.06,0.13,0.78,0.39,0.45,0.71,0.59,0.24,0.41,0.27,0.36,0.4,0.42,0.84,0.53,0.5,0.19,0.96,0.89,0.26,0.35,0.43,0.99,0.39,0.74,0.43,0.62,0.46,0.36,0.35,0.54,0.84,0.29,0.47,0.21,0.32,0.24,0.38,0.63,0.1,0.14,0.23,0.23,0.45,0.64,0.24,0.19,0.32,0.21,0.78,0.01,0.98,0.97,0.98,0.77,0.05,0.63,0.63,0.96,0.07,0.21,0.02,0.18,0.42,0.95,0.83,0.8,0.66,0.14,0.39,0.29,0.92,0.59,0.59,0.95,0.9,0.05,0.83,0.51,0.71,0.48,0.55,0.54,0.77,0.31,0,0.06,0.84,0.9,0.89,0.46,0.69,0.92,0.97,0.59,0.32,0.91,0.38,0.02,0.76,0.01,0.64,0.54,0.86,0.27,0.81,0.42,0.56,0.46,0.64,0.25,0,0.77,0.1,0.03,0.53,0.4,0.99,0.33,0.65,0.36,0.77,0.73,0.64,0.99,0.57,0.92,0.35,0.49,0.99,0.52,0.02,0.82,0.74,0.14,0.92,0.74,0.88,0.72,0.61,0.33,0.51,0.82,0.38,0.46,0.43,0.27,0.34,0.59,0.74,0.7,0.67,0.75,0.13,0.49,0.56,0.37,0.62,0.57,0.39,0.68,0.28,0.26,0.87,0.48,0.23,0.77,0.88,0.1,0.83,0.33,0.91,0.04,0.04,0.24,0.74,0.63,0.88,0.07,0.24,0.52,0.1,0.84,0.05,0.48,0.7,0.76,0.36,0.38,0.25,0.73,0.65,0.45,0.27,0.41,0.61,0.79,0.32,0.21,0.89,0.6,0.01,0.92,0.76,0.82,0.99,0,0.4,0.67,0.59,0.48,0.78,0.3,0.2,1,0.63,0.63,0.7,0.87,0.11,0.03,0.59,0.99,0.36,0.77,0.65,0.46,0.17,0.72,0.26,0.91,0.78,0.82,0.64,0.41,0.5,0.75,0.06,0.02,0.46,0.31,0.68,0.65,0.87,0.21,0.1,0.42,0.96,0.19,0.92,0.53,0.8,0.61,0.54,0.2,0.99,0.61,0.79,0.35,0.48,0.49,0.68,0.4,0.91,0.78,0.56,0.41,0.44,0.55,0.99,0.62,0.52,0.67,0.43,0.89,0.42,0.31,0.78,0.69,0.44,0.71,0.47,0.73,0.09,0.56,0.87,0.39,0.65,0.13,0.41,0,0.46,0.59,0.05,0.34,0.98,0.69,0.35,0.73,0.43,0.17,0,0.86,0.34,0.25,0.18,0.66,0.11,0.51,0,0.71,0.67,0.8,0.1,0.25,0.67,0.49,0.14,0.04,0.99,0.67,0.48,0.83,0.7,0.94,0.74,0.83,0.2,0.27,0.57,0.32,0.87,0.05,0.39,0.7,0.51,0.79,0.22,0.7,0.96,0.89,0.9,0.87,0.17,0.53,0.34,0.87,0.76,0.28,0.79,0.02,0.71,0.5,0.23,0.21,0.76,0.19,0.44,0.81,0.83,0.54,0.31,0.51,0.16,0.63,0.45,0.2,0.96,0.8,0.42,0.9,0.96,0.45,0.17,0.97,0.28,0.65,0.02,0.89,0.37,0.39,0.68,0,0.24,0.84,0.25,0.79,0.95,1,0.35,0.89,0.85,0.18,0.61,0.46,0.07,0.67,0.7,0.08,0.06,0.5,0.99,0.56,0.09,0.03,0.24,0.82,0.22,0.92,0.82,0.68,0.43,0.27,0.06,0.76,0.83,0.09,0.08,0.87,0.12,0.48,0.45,0.46,0.47,0.85,0.1,0.34,0.42,0.43,0.9,0.62,0.44,0.08,0.98,0.17,0.1,0.86,0.59,0.15,0.34,0.77,0.21,0.71,0.6,0.58,0.91,0.57,0.4,0.43,0.56,0.83,0.53,0.62,0.95,0.62,0.29,0.8,0.36,0.45,0.89,0.58,0.89,0.3,0.39,0.75,0.28,0.88,0.64,0.02,0.5,0.01,0.98,0.51,0.7,0.19,0.75,0.66,0.46,0.41,0.81,0.42,0.89,0.11,0.3,0.4,0.12,0.45,0.59,0.14,0.81,0.29,0.79,0.01,0.33,0.51,0.2,0.51,0.5,0.09,0.49,0.65,0.85,0.3,0.17,0.24,1,0.76,0.32,0.44,0.3,0.99,0.29,0.64,0.52,0.32,0.96,0.7,0.86,0.4,0.87,0.67,0.91,0.06,0.61,0.83,0.51,0.86,0.08,0.24,0.76,0.29,0.93,0.64,0.11,0.73,0.08,0.82,0.08,0.74,0.54,0.03,0.85,0.76,0.18,0.13,0.1,0.9,0.88,0.94,0.53,0.24,0.93,0.06,0.67,0.83,0.59,0.53,0.96,0.45,0.5,0.29,0.35,0.35,0.33,0.01,0.71,0.25,0.22,0.82,0.72,0.13,0.42,0.11,0.24,0.55,0.65,0.04,0.12,0.35,0.64,0.27,0.08,0.22,0.5,0.56,0.23,0.3,0.14,0.32,0.36,0.57,0.51,0.82,0.68,0.6,0.76,0.5,0.62,0.43,0.65,0.67,0.96,0.3,0.3,0.77,0.82,0.99,0.33,0.5,0.03,0.64,0.16,0.43,0.1,0.17,0.91,0.32,0.85,0.65,0.74,0.75,0.96,0.44,0.58,0.93,0.36,0.12,0.65,0.94,0.89,0.79,0.64,0.28,0.7,0.88,0,0.64,0.28,0.43,0.49,0.49,0.52,0.41,0.62,0.5,0.58,0.21,0.38,0.56,0.72,0.05,0.42,0.41,0.03,0.02,0.26,0.94,0.23,0.33,0.65,0.61,0.24,0.12,0.07,0.34,0.27,0.07,0.76,0.38,0.66,0.63,0.35,0.92,0.7,0.19,0.88,0.6,0.86,0.47,0.95,0.6,0.19,0.94,0.34,0.64,0.32,0.5,0.02,0.6,0.47,0.48,0.67,0.98,0.34,0.72,0.27,0.95,0.87,0.1,0.17,0.9,0.07,0.69,0.96,0.63,0.4,0.39,0.21,0.9,0.42,0.63,0.86,0.62,0.77,0.03,0.75,0.08,0.55,0.83,0.25,0.76,0.27,0.29,0.43,0.03,0.91,0.06,0.53,0.12,0.66,0.26,0.6,0.15,0.3,0.33,0.98,0.57,0.62,0.67,0.28,0.19,0.15,0.03,0.08,0.95,0.8,0.35,0.12,0.96,0.88,0.78,0.36,0.27,0.61,0.83,0.49,0.97,0.66,0.52,1,0.56,0.05,0.36,0.15,0,0.68,0.55,0.25,0.1,0.95,0.57,0.25,0.63,0.89,0.99,0.68,0.78,0.23,0.08,0.75,0.5,0.46,0.08,0.34,0.66,0.14,0.94,0.98,0.74,0.01,0.62,0.86,0.51,0.41,0.91,0.14,0.9,0.83,0.02,0.53,0.85,0.96,0.5,0.5,0.09,0.53,0.85,0.49,0.79,0.72,0.56,0.2,0.51,0.06,0.08,0.78,0.48,0.69,0.5,0.48,0.74,0.46,0.37,0.62,0.3,0.83,0.53,0.8,0.97,0.51,0.19,0.62,0.79,1,0.93,0.23,0.19,0.95,0.72,0.26,0.35,0.74,0.16,0.57,0.21,0.22,0.9,0.63,0.85,0.87,0.06,0.11,0.51,0.96,0.26,0.46,0.16,0.5,0.17,0.36,0.17,0.29,0.25,0.71,0.33,0.06,0.48,0.68,0.87,0.7,0.08,0.95,0.79,0.03,0.76,0.55,0.26,0.56,0.32,0.58,0.52,0.83,0.08,0.9,0.26,0.31,0.51,0.76,0.28,0.77,0.09,0.75,0.19,0.95,0.97,0.63,0.1,0.74,0.26,0.72,0.1,0.54,0.18,0.71,0.94,0.07,0.67,0.82,0.49,0.32,0.92,0.22,0.46,0.81,0.25,0.32,0.18,0.38,0.94,0.39,0.59,0.88,0.24,0.08,0.13,1,0.27,0.02,0.52,0.17,0.57,0.96,0.18,0.45,0.62,0.6,0.58,0.82,0.24,0.24,0.59,0.1,0.11,0.3,0.37,0.05,0.38,0.88,0.74,0.39,0.67,0.18,0.29,0.17,0.17,0.28,0.34,0.02,0.37,0.97,0.65,0.8,0.77,0.36,0.86,0.06,1,0.49,0.73,0.78,0.85,0.24,0.29,0.25,0.39,0.34,0.03,0.94,0.78,0.26,0.62,0.41,0.12,0.47,0.72,0.08,0.44,0.66,0.03,0.35,0.59,0.28,0.26,0.89,0.69,0.94,0.4,0.1,0.87,0.65,0.74,0.11,0.43,0.15,0.28,0.79,0.15,0.49,0.45,0.73,0.57,0.17,0.37,0.61,0.43,0.34,0.97,0.54,0.52,0.49,0.14,0.57,0.37,0.96,0.72,0.27,0.47,0.05,0.8,0.03,0.82,0.76,0.02,0.15,0.75,0.71,0.54,0.27,0.79,0.68,0.16,0,0.53,0.45,0.55,0.51,0.92,0.27,0.17,0.47,0.38,0.47,0.42,0.93,0.91,0.33,0.01,0.82,1,0.54,0.25,0.67,0.52,0.43,0.78,0.78,0.81,0.54,0.21,0.23,0.2,0.76,0.04,0.37,0.22,0.35,0.96,0.58,0.48,0.63,0.54,0.57,0.63,0.29,0.5,0.47,0.73,0.42,0.28,0.43,0.36,0.16,0.06,0.66,0.19,0.47,0.48,0.68,0.06,0.23,0.38,0.73,0.62,0.34,0.4,0.91,0.67,0.62,0,0.43,0.19,0.89,0.85,0.32,0.65,0.36,0.33,0.6,0.87,0.89,0.94,0.18,0.23,0.3,0.8,0.43,0.05,0.21,0.95,0.55,0.37,0.14,0.28,0.74,0.05,0.71,0.09,0.05,0.26,0.5,0.02,0.65,0.6,0.33,0.7,0.85,0.78,0.78,0.15,0.5,0.35,0.58,0.44,0.54,0.49,0.17,0.96,0.75,0.05,0.92,0.04,0.14,0.76,0.09,0.61,0.3,0.89,0.18,0.56,0.03,0.9,0.57,0.38,0.77,0.68,0.55,0.79,0.77,0.64,0.59,0.64,0.34,0.43,0.22,0.75,0.24,0.66,0.73,0.11,0.71,0.66,0.8,0.44,0.53,0.67,0.53,0.38,0,0.33,0.55,0.9,0.39,0.41,0.08,0.88,0.26,0.54,0.53,0.28,0.14,0.09,0.49,0.16,0.87,0.45,0.18,0.06,0.11,0.07,0.11,0.29,0.48,0.44,0.03,0.58,0.17,0.44,0.81,0.74,0.62,0.83,0.16,0.62,0.13,0.85,0.58,0.38,0.9,0.58,0.04,0.81,0.98,0.11,0.57,0.96,0.87,0.71,0.28,0.62,0.67,0.94,0.02,0.71,0.21,0.73,0.9,0.95,0.31,0.8,0.31,0.73,0.46,0.88,0.75,0.69,0.88,0.86,0.28,0.11,0.17,0.72,0.48,0.1,0.83,0.05,0.27,0.35,0.15,0.35,0.01,0.44,0.14,0.99,0.39,0.74,0.17,0.28,0.56,0.37,0.78,0.5,0.85,0.61,0.21,0.17,0.35,0.83,0.13,0.41,0.8,0.69,0.34,0.46,0.05,0.11,0.01,0.1,0.59,0.96,0.53,0.04,0.5,0.86,0.75,0.91,0.36,0.1,0.53,0.5,0.86,0.12,0.01,0.51,0.06,0.19,0.65,0.01,0.92,0.62,0.08,0.05,0.78,0.86,0.01,0.8,0.07,0.04,0.03,0.16,0.52,0.93,0.06,0.99,0.31,0.8,0.46,0.21,0.99,0.37,0.1,0.22,0.63,0.5,0.05,0.88,0.45,0.92,0.08,0.78,0.67,0.31,0.18,0.01,0.13,0.84,0.57,0.17,0.31,0.69,0.82,0.69,0.19,0.51,0.49,0.53,0.99,0.43,0.45,0.3,0.19,0.43,0.16,0.11,0.16,0.54,0.65,0.4,0.35,0.78,0.59,0.14,0.54,0.26,0.3,0.56,0.96,0.1,0.85,0.3,0.63,0.08,0.35,0.18,0.36,0.3,0.55,0.09,0.65,0.68,0.65,0.67,0.64,0.31,0.84,0.17,0.7,0.56,0.85,0.64,0.25,0.71,0.04,0.65,0.3,0.27,0.12,0.15,0.64,0.37,0.84,0.3,0.93,0.55,0.97,0.14,0.74,0.72,0.32,0.17,0.25,0.34,0.43,0.58,0.72,0.9,0.05,0.95,0.31,0.19,0.26,0.35,0.19,0.1,0.71,0.07,0.49,0.73,0.81,0.46,0.07,0.22,0.37,0.04,0.56,0.13,0.85,0.75,0.86,0.89,0.16,0.39,0.32,0.9,0.2,0.15,0.89,0.51,0.27,0.03,0.8,0.21,0.03,0.94,0.67,0.01,0.78,0.31,0.18,0.36,0.27,0.46,0.32,0.08,0.98,0.1,0.26,0.01,0.9,0.31,0.56,0.65,0.52,0.59,0.42,0.77,0.47,0.62],
        topK: 1,
        includeMetadata: true,
        includeValues: true,
        filter: {
          "FAMILYIDNO": {"$eq": record.userData.FAMILYIDNO},
          "FATHER_NAME_ENG": {"$eq": record.userData.FATHER_NAME_ENG},
          "HOF_NAME_ENG": {"$eq": record.userData.HOF_NAME_ENG},
        }
      })
      const matchingDocInPinecone = await index.query({
        vector: document.matches[0].values,
        topK: 2,
        includeMetadata: true
      })
      const matchingDocInDB = await DummyData.findOne({"userData.FAMILYIDNO": matchingDocInPinecone.matches[1].metadata.FAMILYIDNO, "userData.FATHER_NAME_ENG": matchingDocInPinecone.matches[1].metadata.FATHER_NAME_ENG, "userData.HOF_NAME_ENG": matchingDocInPinecone.matches[1].metadata.HOF_NAME_ENG})

      record.pineconeId = document.matches[0].id
      record.mostSimilarDocument = matchingDocInDB._id,
      record.similarityScore = Math.round(matchingDocInPinecone.matches[1].score * 100)
      await record.save()
      console.log("Record Updated", i + 1)
    }
  }

  async totalDocumentsInDB() {
    return await DummyData.find()
  }

  async totalDuplicateDocumentsInDB() {
    return await DummyData.find({status: 'duplicate'})
  }

  async changeStatus(status, docId) {
    await DummyData.findByIdAndUpdate(docId, {
      status: status
    })
  }

  async getOneRecord(recordId) {
    const record = await DummyData.findById(recordId).populate("mostSimilarDocument")
    return {
      document: record.userData,
      duplicate: record.mostSimilarDocument.userData
    }
  }
}

module.exports = new GoogleService();
