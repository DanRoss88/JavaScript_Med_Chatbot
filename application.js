const natural = require("natural");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

const { layers, Sequential } = require("@tensorflow/tfjs-node");
const { tensor } = tf;

const stemmer = natural.PorterStemmer.stem;
const tokenizer = new natural.WordTokenizer();

const data = JSON.parse(fs.readFileSync("./input/intents.json", "utf8"));

let words = [];
let labels = [];
let docs_x = [];
let docs_y = [];

data.intents.forEach((intent) => {
  intent.patterns.forEach((pattern) => {
    const wrds = pattern.split(" ");
    words.push(...wrds);
    docs_x.push(wrds);
    docs_y.push(intent.tag);
  });

  if (!labels.includes(intent.tag)) {
    labels.push(intent.tag);
  }
});

words = words.filter((w) => w !== "?").map((w) => stemmer(w.toLowerCase()));
words = [...new Set(words)].sort();
labels.sort();

const training = [];
const output = [];
const outEmpty = Array(labels.length).fill(0);

docs_x.forEach((doc, x) => {
  const bag = [];

  const wrds = doc.map((w) => stemmer(w));

  words.forEach((w) => {
    if (wrds.includes(w)) {
      bag.push(1);
    } else {
      bag.push(0);
    }
  });

  const outputRow = [...outEmpty];
  outputRow[labels.indexOf(docs_y[x])] = 1;

  training.push(bag);
  output.push(outputRow);
});

const X = tensor(training);
const y = tensor(output);

const model = tf.sequential();
model.add(layers.dense({ units: 8, inputShape: [X.shape[1]] }));
model.add(layers.dense({ units: 8 }));
model.add(layers.dense({ units: output[0].length, activation: "softmax" }));
model.compile({
  loss: "categoricalCrossentropy",
  optimizer: "adam",
  metrics: ["accuracy"],
});

(async () => {
  try {
     model = await tf.loadLayersModel('file://model/model.json');
  } catch (err) {
    await model.fit(X, y, { epochs: 1000, batchSize: 8 });
    await model.save("file://model");
  }

  function bagOfWords(s, words) {
    const bag = Array(words.length).fill(0);

    const sWords = tokenizer.tokenize(s).map((w) => stemmer(w.toLowerCase()));

    sWords.forEach((se) => {
      const index = words.indexOf(se);
      if (index !== -1) {
        bag[index] = 1;
      }
    });

    return tensor([bag]);
  }

  function chat() {
    console.log("Start Talking with the bot (type quit to stop!)");
    const readline = require("readline");
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    rl.on("line", async (input) => {
      if (input.toLowerCase() === "quit") {
        rl.close();
        return;
      }

      const results = await model.predict(bagOfWords(input, words)).data();
      const resultsIndex = Array.from(results).indexOf(Math.max(...results));
      const tag = labels[resultsIndex];

      console.log("Detected Tag:", tag); // Print the detected tag

      if (results[resultsIndex] > 0.5) {
        const tg = data.intents.find((intent) => intent.tag === tag);
        const responses = tg.responses;
        console.log(
          "Bot Response:",
          responses[Math.floor(Math.random() * responses.length)]
        ); // Print the bot's response
        console.log("\n");
      } else {
        console.log("I didn't get that, try again");
      }
    });
  }

  chat();
})();
