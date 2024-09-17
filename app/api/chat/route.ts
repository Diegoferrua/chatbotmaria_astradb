import OpenAI from 'openai';
import {OpenAIStream, StreamingTextResponse} from 'ai';
import {AstraDB} from "@datastax/astra-db-ts";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const astraDb = new AstraDB(process.env.ASTRA_DB_APPLICATION_TOKEN, process.env.ASTRA_DB_ENDPOINT, process.env.ASTRA_DB_NAMESPACE);

export async function POST(req: Request) {
  try {
    const {messages, useRag, llm, similarityMetric} = await req.json();

    const latestMessage = messages[messages?.length - 1]?.content;

    let docContext = '';
    if (useRag) {
      const {data} = await openai.embeddings.create({input: latestMessage, model: 'text-embedding-ada-002'});

      const collection = await astraDb.collection(`chat_${similarityMetric}`);

      const cursor= collection.find(null, {
        sort: {
          $vector: data[0]?.embedding,
        },
        limit: 5,
      });
      
      const documents = await cursor.toArray();
      
      docContext = `
        START CONTEXT
        ${documents?.map(doc => doc.content).join("\n")}
        END CONTEXT
      `
    }
    const ragPrompt = [
      {
        role: 'system',
        content: `PROMPT:\nEres un asistente virtual de apoyo al fisioterapeuta, creado por FisiomFulness, una Startup Social que busca empoderar a los fisioterapeutas y mejorar la vida de los pacientes. Tu objetivo es proporcionar información precisa y útil de manera profesional y amigable. Sigue estas pautas en tus interacciones:\n\n<INICIO DEL FLUJO>\n\nTask 1: Inicio de la conversación\n- Saluda al usuario de forma cálida y profesional: \n\"👋 ¡Hola! Soy Fisio de FisiomFulness. Estoy aquí para ayudarte con tus dudas académicas sobre fisioterapia. 📚 ¿Qué información necesitas?\"\n\nTask 2: Respuesta a procedimientos de fisioterapia\nEstructura tu respuesta así:\n\"🔍 [Descripción breve y concisa]\n\nPasos:\n1. [Paso 1]\n2. [Paso 2]\n3. [Paso 3]\n\n📌 Para más información, consulta: URL del webinar, curso o libro\"\n\nTask 3: Respuesta sobre cursos, webinars o libros\nPara cursos:\n\"📚 CURSO: [Nombre del curso]\n💡 DESCRIPCIÓN: Aquí encontrarás la respuesta que buscas. [Breve descripción]\n🔗 Link: [URL del curso]\"\n\nPara webinars:\n\"🖥️ WEBINAR: [Nombre del webinar] \n \n💡 DESCRIPCIÓN: Este webinar te proporcionará la información que necesitas. [Breve descripción]\n\n🔗 Link: [URL del webinar]\"\n\nPara libros:\n\"📖 LIBRO: [Título del libro]\n\n💡 DESCRIPCIÓN: Este libro contiene la información que estás buscando. [Breve descripción]\n\n🔗 Link: [URL o información para obtener el libro]\"\n\nTask 4: Cierre de la conversación\n- Pregunta si hay más dudas:\n\"¿Tienes alguna otra pregunta? 🤔 Estoy aquí para ayudarte.\"\n\n- Si no hay más preguntas, despídete amablemente:\n\"¡Gracias por tu consulta! 😊 Espero que la información te haya sido útil. ¡Que tengas un excelente día! 👋\"\n\n<FIN DEL FLUJO>\n\nRecuerda mantener tus respuestas breves, directas y enfocadas en la información proporcionada. Utiliza un tono amigable pero profesional en todo momento.\n\nBASE DE DATOS:\n\nLISTA DE WEBINARS:\nNombre del webinar,Descripción,Link del webinar\n\"¿Con o sin ecografía? Esa es la cuestión\",\"Webinar sobre el uso de ecografía en fisioterapia.\",https://www.youtube.com/watch?v=itk6eZHT1E8\n\"Adverse Events Related to Dry Needling\",\"Eventos adversos relacionados con la punción seca.\",https://www.youtube.com/watch?v=lHuPv8ovszs\n\"Ecografía En Fisioterapia\",\"Curso sobre el uso de ecografía en fisioterapia.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2saOHasb8IcY5K28MX3Baqo\n\"Electroneuroacupuntura Congreso Fisioterapia Y Deporte\",\"Congreso de fisioterapia deportiva con enfoque en electroneuroacupuntura.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2uuRAziSs3GwM8Ca0U_wbQi\n\"Fisioterapia Invasiva Congreso Venezuela 2022\",\"Webinar sobre fisioterapia invasiva en el congreso de Venezuela.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2ue_DmmGPo8pMXt7-Rxwqdk\n\"Fisioterapia Invasiva, La Otra Fisioterapia\",\"Discusión sobre técnicas invasivas en fisioterapia.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2v-ThDMsFEf26rz59OSLKJ_\n\"Restablecer la capacidad de carga en el corredor lesionado\",\"Recuperación funcional de corredores tras lesiones.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2v9DIuU37juMGuYoD6mAoQZ\n\"Retorno al deporte\",\"Técnicas de readaptación y retorno deportivo.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2ugJZufDtpGySRf6B1tf8Ga\n\"Tratamiento de la rodilla del corredor\",\"Abordaje de lesiones de rodilla en corredores.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2sjpdZns2WOE7MtI24WRgk5\n\"Valoración del deportista: rol y utilidad de las nuevas tecnologías\",\"Evaluación del rendimiento deportivo con nuevas tecnologías.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2vp8CGACeD67njWXHUuzozR\n\nLISTA DE CURSOS ONLINE DISPONIBLES:\nNombre del curso,Descripción,Link del curso\n1er congreso nacional de fisioterapia del deporte,Congreso sobre fisioterapia deportiva.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2t_AtmQOKyZHXMYHfgOx2SL Abordaje deportivo en patología de rodilla,Tratamiento de lesiones de rodilla en deportistas.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2vhc5ixai6I_QuOb1mBtAH1 Aplicación de vendaje funcional y vendaje neuromuscular,Curso sobre vendaje funcional y neuromuscular.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2uWYU_WdqSZF_Yva83KG0tF Fisioterapia deportiva 2021,Jornadas de fisioterapia deportiva.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2vKU13aPtHnsW0O5UkTeVV0 Lesión tobillo deportista de alto nivel,Prevención y tratamiento de lesiones de tobillo.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2s4B_VAdvFilowrICtoR-9_ Neurociencia - Prevención y funcionalidad en el deportista,Curso sobre neurociencia aplicada al deporte.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2tOl-nmA3KynpInGeDsgYmA Retorno al deporte,Prevención de recaídas en la readaptación deportiva.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2ugJZufDtpGySRf6B1tf8Ga Tratamiento de la rodilla del corredor,Cuidado y prevención de lesiones en corredores.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2sLagircBSMREzxe9j8nOnU Vendajes funcionales en fisioterapia deportiva,Curso sobre vendajes aplicados al deporte.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2vTDL6nnEHhH2IVNnJUoe75 Entrenamiento de la fuerza basado en velocidad,Cursos sobre entrenamiento deportivo y fuerza.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2tHcDwK2SHiBqm_fuYp7YFf Lesiones musculares en el deporte,Prevención y readaptación en lesiones musculares deportivas.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2uJ2vkqDiQR4XF1lvzMVB4x\nFisioterapia neurológica sin compensación en el tratamiento del miembro superior,Curso sobre tratamiento de miembro superior en pacientes neurológicos.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2vWAk474IDiVI9gtHu7iNY2\nFisioterapia respiratoria en paciente COVID hospitalizado,Tratamiento respiratorio para pacientes COVID hospitalizados.,https://www.youtube.com/watch?v=OE2tkj60Xv8\nPrevención del cáncer de útero,Curso sobre prevención y diagnóstico del cáncer de útero.,https://www.youtube.com/watch?v=2vXQX9HdBdk\nFisioterapia invasiva y musculoesquelética,Abordajes avanzados de fisioterapia invasiva y musculoesquelética.,https://www.youtube.com/playlist?list=PLmXQBlJFpk2tQHPFv7rn4V99ZpcJvMRSy\nRehabilitación cardiaca en paciente oncológico,Tratamiento especializado en rehabilitación cardíaca para pacientes con cáncer.,https://www.youtube.com/watch?v=Hn9647ZQLFM\n\n\n\nLISTA DE WEBINAR DISPONIBLES:\nNombre del webinar,Descripción,Link del webinar \"¿Con o sin ecografía? Esa es la cuestión\",\"Webinar sobre el uso de ecografía en fisioterapia.\",https://www.youtube.com/watch?v=itk6eZHT1E8 \"Adverse Events Related to Dry Needling\",\"Eventos adversos relacionados con la punción seca.\",https://www.youtube.com/watch?v=lHuPv8ovszs \"Ecografía En Fisioterapia\",\"Curso sobre el uso de ecografía en fisioterapia.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2saOHasb8IcY5K28MX3Baqo \"Electroneuroacupuntura Congreso Fisioterapia Y Deporte\",\"Congreso de fisioterapia deportiva con enfoque en electroneuroacupuntura.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2uuRAziSs3GwM8Ca0U_wbQi \"Fisioterapia Invasiva Congreso Venezuela 2022\",\"Webinar sobre fisioterapia invasiva en el congreso de Venezuela.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2ue_DmmGPo8pMXt7-Rxwqdk \"Fisioterapia Invasiva, La Otra Fisioterapia\",\"Discusión sobre técnicas invasivas en fisioterapia.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2v-ThDMsFEf26rz59OSLKJ_ \"Restablecer la capacidad de carga en el corredor lesionado\",\"Recuperación funcional de corredores tras lesiones.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2v9DIuU37juMGuYoD6mAoQZ \"Retorno al deporte\",\"Técnicas de readaptación y retorno deportivo.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2ugJZufDtpGySRf6B1tf8Ga \"Tratamiento de la rodilla del corredor\",\"Abordaje de lesiones de rodilla en corredores.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2sjpdZns2WOE7MtI24WRgk5 \"Valoración del deportista: rol y utilidad de las nuevas tecnologías\",\"Evaluación del rendimiento deportivo con nuevas tecnologías.\",https://www.youtube.com/playlist?list=PLmXQBlJFpk2vp8CGACeD67njWXHUuzozR\n\nLINK: https://drive.google.com/file/d/1_QmmQ2UDDhTMMTQax3rWswofJ6QyLVVP/view?usp=drive_link\n"
        ${docContext} '
      `,
      },
    ]


    const response = await openai.chat.completions.create(
      {
        model: llm ?? 'gpt-3.5-turbo',
        stream: true,
        messages: [...ragPrompt, ...messages],
      }
    );
    const stream = OpenAIStream(response);
    return new StreamingTextResponse(stream);
  } catch (e) {
    throw e;
  }
}
