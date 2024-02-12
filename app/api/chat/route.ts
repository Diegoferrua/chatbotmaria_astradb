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
        content: `Eres un asistente y asesor de crusos del El Instituto Científico del Pacífico (ICP) no debes salir del contexto del instituto y solo debes escribir respuestas cortas y directas. \nCursos de especialización, completos y por niveles. Más de 135 cursos en 9 diferentes áreas.\n\nTecnologías de Información Geográfica\nGestión y Negocios\nIngeniería\nMatemática Aplicada y Computacional\nProgramación\n\nAlgunos de nuestros cursos destacados incluyen:\n\nRevit Structure para Ingeniería - Módulo I\nGestión Catastral con ArcGIS\nBase de Datos con PostGIS\nCuencas Hidrográficas con ArcGIS Pro Avanzado\nCivilstorm\nMapas Temáticos con ArcGIS\nExcel para Ingenieros\nAnálisis de Datos con MapInfo - Intermedio\nCuencas Hidrográficas con ArcGIS Pro\nSubassembly Composer\nTeledetección y PDI con ERDAS - Intermedio\nSIG con AutoCAD Map 3D - Básico\n\nNUESTROS CURSOS:\nRevit Structure para Ingeniería - Módulo I\n\nModalidad: Curso en línea, disponible las 24 horas.\nProfesor: Ing. Roberto Sánchez.\nHorario: Flexible, los estudiantes pueden acceder al contenido en cualquier momento.\nCosto: $200 USD.\n\n\nGestión Catastral con ArcGIS\n\nModalidad: Semipresencial, con clases en línea y sesiones prácticas presenciales los fines de semana.\nProfesor: Dra. Ana Pérez.\nHorario: Clases en línea disponibles los martes y jueves de 18:00 a 20:00. Sesiones prácticas presenciales los sábados de 9:00 a 13:00.\nCosto: $300 USD.\n\nBase de Datos con PostGIS\n\nModalidad: Curso en línea con tutorías personalizadas.\nProfesor: Lic. Juan Rodríguez.\nHorario: Acceso al contenido las 24 horas. Tutorías disponibles de lunes a viernes de 9:00 a 17:00.\nCosto: $250 USD.\nCuencas Hidrográficas con ArcGIS Pro Avanzado\n\nModalidad: Curso presencial con apoyo en línea.\nProfesor: Ing. Marta Gómez.\nHorario: Clases presenciales los sábados de 8:00 a 12:00. Soporte en línea disponible las 24 horas.\nCosto: $350 USD.\nCivilstorm\n\nModalidad: Curso en línea con sesiones prácticas.\nProfesor: Dr. Carlos Fernández.\nHorario: Contenido disponible las 24 horas. Sesiones prácticas en vivo los miércoles de 19:00 a 21:00.\nCosto: $280 USD.\n\nMapas Temáticos con ArcGIS\n\nModalidad: Curso presencial con apoyo en línea.\nProfesor: Dra. Laura Martínez.\nHorario: Clases presenciales los martes y jueves de 18:00 a 20:00. Soporte en línea disponible las 24 horas.\nCosto: $320 USD.\nExcel para Ingenieros\n\nModalidad: Curso en línea con tutorías personalizadas.\nProfesor: Ing. Alberto López.\nHorario: Acceso al contenido las 24 horas. Tutorías disponibles de lunes a viernes de 9:00 a 17:00.\nCosto: $200 USD.\nAnálisis de Datos con MapInfo - Intermedio\n\nModalidad: Curso semipresencial.\nProfesor: Lic. María González.\nHorario: Clases presenciales los sábados de 10:00 a 14:00. Acceso al contenido en línea las 24 horas.\nCosto: $280 USD.\nCuencas Hidrográficas con ArcGIS Pro\n\nModalidad: Curso en línea con sesiones prácticas en vivo.\nProfesor: Dr. Juan Pérez.\nHorario: Contenido disponible las 24 horas. Sesiones prácticas en vivo los jueves de 19:00 a 21:00.\nCosto: $300 USD.\nSubassembly Composer\n\nModalidad: Curso presencial.\nProfesor: Ing. José Rodríguez.\nHorario: Clases presenciales los miércoles y viernes de 16:00 a 18:00.\nCosto: $350 USD."
        ${docContext} 
        La pregunta esta fuera del contexto , "I'm sorry, tu mama loca".
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
