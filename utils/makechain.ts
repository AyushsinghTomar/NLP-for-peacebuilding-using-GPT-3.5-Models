import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question. Provide a conversational answer and explanations based on the context provided.
You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks. Use different chunks when answering.
If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
You will follow the following process to answer:
1.Based on my input, you create 4 sections for the output.
a. You will add a title of the context in bold.
b. You restate the question in a sentence in italics.
c. You will provide with two suggestions for next questions in a list form.
Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.


Question: {question}
=========
{context}
=========
Answer in Markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 1.0, // increase temepreature to get more creative answers
    modelName: 'text-davinci-003', //change this to gpt-4 if you have access text-davinci-003 gpt-3.5-turbo
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
