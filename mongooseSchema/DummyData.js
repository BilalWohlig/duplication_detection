const mongoose = require('mongoose')
const Schema = mongoose.Schema

const schema = new Schema({
    userData: {
        type: Object
    },
    similarityScore: {
        type: Number
    },
    mostSimilarDocument: {
        type: Schema.Types.ObjectId,
        ref: "DummyData"
    },
    pineconeId: {
        type: String
    },
    status: {
        type: String,
        enum: ['duplicate', 'non-duplicate']
    }
},
{ timestamps: true })
module.exports = mongoose.model('DummyData', schema)
