const axios = require('axios')
const { Pinecone } = require('@pinecone-database/pinecone')

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_GOOGLE_API_KEY
})
const { Configuration, OpenAIApi } = require('openai')
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY
})
const openai = new OpenAIApi(configuration)
const { v4: uuidv4 } = require('uuid')
const index = pinecone.index('google-data')
class GoogleService {
  async pushDataToPinecone (data) {
    const batch_size = 100
    for (let i = 0; i < data.length; i += batch_size) {
      const i_end = Math.min(data.length, i + batch_size)
      const meta_batch = data.slice(i, i_end)
      const ids_batch = meta_batch.map((x) => {
        return uuidv4()
      })
      const texts_batch = meta_batch.map((x) => {
        let stringData = Object.values(x)
        stringData = stringData.join(', ')
        return stringData
      })
      let response
      try {
        response = await openai.createEmbedding({
          model: 'text-embedding-3-large',
          input: texts_batch
        })
      } catch (error) {
        console.log('Error while creating embedding', error)
      }
      const embeds = response.data.data.map((record) => record.embedding)
      const meta_batch_cleaned = meta_batch.map((x) => ({
        FAMILYIDNO: x.FAMILYIDNO,
        AADHAR_REF_IDL: x.AADHAR_REF_ID,
        HOF_NAME_ENG: x.HOF_NAME_ENG,
        FATHER_NAME_ENG: x.FATHER_NAME_ENG,
        MOTHER_NAME_ENG: x.MOTHER_NAME_ENG,
        DOB: x.DOB,
        ALL_DATA: Object.values(x).join(', ')
      }))
      const to_upsert = ids_batch.map((id, i) => ({
        id: id,
        values: embeds[i],
        metadata: meta_batch_cleaned[i]
      }))
      await index.upsert(to_upsert)
      console.log('Successfully uploaded')
    }
  }

  async getDataFromPinecone (data) {
    let stringData = Object.values(data)
    stringData = stringData.join(', ')
    const embeddings = await openai.createEmbedding({
      model: 'text-embedding-3-large',
      input: stringData
    })
    const embeds = embeddings.data.data.map((record) => record.embedding)
    const response = index.query({
      vector: embeds[0],
      topK: 5,
      includeValues: false,
      includeMetadata: true
    })
    return response
  }
}

module.exports = new GoogleService()
