const mongoose = require('mongoose')
const Schema = mongoose.Schema

const schema = new Schema({
    userData: {
        type: Object
    }
},
{ timestamps: true })
module.exports = mongoose.model('DummyData', schema)
