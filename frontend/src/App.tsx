import { useState } from "react";

import "./App.css";

type ActionItem = {
  owner: string;
  task: string;
  deadline: string;
};

type MeetingResult = {
  summary: string;
  decisions: string[];
  action_items: ActionItem[];
};

// 1. Define an interface for your backend API response
interface uploadResponse {
  success: boolean;
  message?: string;
  fileUrl?: string;
   
}


function App() {
  const [notes, setNotes] = useState("");
  const [result, setResult] = useState<MeetingResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);


  /**
 * Uploads a file asynchronously to a specified API endpoint.
 * @param file - The native browser File object to upload.
 * @param endpoint - The backend API destination URL.
 * @returns A promise resolving to the typed backend response.
 */

  async function uploadFile(
  file: File,
  endpoint: string
): Promise<uploadResponse> {
  const formData = new FormData();

  formData.append("file", file);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Upload failed with status: ${response.status}`);
    }

    const data: uploadResponse = await response.json();

    return data;
  } catch (error) {
    console.error("Error uploading file:", error);

    return {
      success: false,
      message: (error as Error).message,
    };
  }
}
  async function handleUpload() {
    if (!selectedFile) {
      return;
    }

    const response = await uploadFile(
      selectedFile,
      "http://127.0.0.1:8000/upload"
    );

    console.log("Backend response:", response);
  }

  return (
    <main className="app-container">

      <header className="app-header">
        <h1>Pdf Q&A Assistant</h1>
        <p>
          Ask questions about your PDF documents and get answers in real-time. Simply upload your PDF, and start asking questions!
        </p>
      </header>

      <section className="input-section">

        <input
          type =  "file"
          accept = ".pdf"
          onChange = {(event) => {
            const file = event.target.files?.[0];

            if(file) {
              setSelectedFile(file);
            }
          }}
          />
          {selectedFile && (
            <p>Selected file: {selectedFile.name}</p>
          )}

          <button
            onClick={handleUpload}
            disabled={!selectedFile}
          >
            Upload PDF
          </button>

        

      </section>

      {result && (
        <section className="results-section">

          <div className="card summary-card">
            <h2>Summary</h2>

            <p className="summary-text">
              {result.summary}
            </p>
          </div>


          <div className="card">
            <h2>Decisions</h2>

            <ul>
              {result.decisions.map((decision, index) => (
                <li
                  key={index}
                  className="decision-item"
                >
                  {decision}
                </li>
              ))}
            </ul>
          </div>


          <div className="card">
            <h2>Action Items</h2>

            <ul>
              {result.action_items.map((item, index) => (
                <li
                  key={index}
                  className="action-item"
                >
                  <span className="action-owner">
                    {item.owner}
                  </span>

                  <span className="action-task">
                    {item.task}
                  </span>

                  <span className="action-deadline">
                    {item.deadline || "No deadline"}
                  </span>
                </li>
              ))}
            </ul>
          </div>

        </section>
      )}

    </main>
  );
}

export default App;