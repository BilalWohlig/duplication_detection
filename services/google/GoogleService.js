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
}

module.exports = new GoogleService();
