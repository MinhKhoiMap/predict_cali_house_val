const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const api = require("./routes/api");
const multer = require("multer");
const upload = multer();

const app = express();
app.use(express.json());
app.use(cors());
app.use(bodyParser.json());
app.use(
  bodyParser.urlencoded({
    extended: true,
  })
);
app.set("view engine", "ejs");

app.use("", api);

app.listen(3000, () => {
  console.log("app ready");
});

module.exports = app;
