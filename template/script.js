
const $ = (id) => document.getElementById(id);

let dataset = null;
let toastTimer;

function notify(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");

  clearTimeout(toastTimer);

  toastTimer = setTimeout(() => {
    el.classList.remove("show");
  }, 2600);
}

// NAVIGATION
function showPage(name) {
  document.querySelectorAll(".page").forEach(page => {
    page.classList.remove("active-page");
  });

  document.querySelectorAll(".nav-link[data-page]").forEach(button => {
    button.classList.toggle("active", button.dataset.page === name);
  });

  $("page-" + name).classList.add("active-page");

  $("crumb").textContent =
    document.querySelector(`[data-page="${name}"]`).textContent.trim();
}

document.querySelectorAll(".nav-link[data-page]").forEach(button => {
  button.addEventListener("click", () => {
    showPage(button.dataset.page);
  });
});

// CSV UPLOAD
$("go-upload").addEventListener("click", () => {
  $("csv-file").click();
});

$("dataset-upload-btn").addEventListener("click", () => {
  $("csv-file").click();
});

$("csv-file").addEventListener("change", event => {
  const file = event.target.files[0];

  if (file) {
    loadCSV(file);
  }

  event.target.value = "";
});

// DRAG AND DROP
const dropzone = $("dropzone");

["dragenter", "dragover"].forEach(eventName => {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach(eventName => {
  dropzone.addEventListener(eventName, event => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});

dropzone.addEventListener("drop", event => {
  const file = event.dataTransfer.files[0];

  if (file) {
    loadCSV(file);
  }
});

// CSV PARSER
function parseCSV(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];

    if (ch === '"' && quoted && next === '"') {
      cell += '"';
      i++;
    }

    else if (ch === '"') {
      quoted = !quoted;
    }

    else if (ch === "," && !quoted) {
      row.push(cell);
      cell = "";
    }

    else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && next === "\n") {
        i++;
      }

      row.push(cell);

      if (row.some(value => value.trim() !== "")) {
        rows.push(row);
      }

      row = [];
      cell = "";
    }

    else {
      cell += ch;
    }
  }

  row.push(cell);

  if (row.some(value => value.trim() !== "")) {
    rows.push(row);
  }

  if (rows.length < 2) {
    throw new Error(
      "The CSV must contain a header and at least one data row."
    );
  }

  const headers = rows[0].map((header, index) => {
    return header.trim() || `Column ${index + 1}`;
  });

  const records = rows.slice(1).map(record => {
    return headers.map((_, index) => {
      return (record[index] ?? "").trim();
    });
  });

  return { headers, records };
}

// LOAD CSV
async function loadCSV(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    return notify("Please upload a .csv file.");
  }

  if (file.size > 20 * 1024 * 1024) {
    return notify("File size must be below 20 MB.");
  }

  try {
    const parsed = parseCSV(await file.text());

    dataset = {
      ...parsed,
      name: file.name,
      size: file.size
    };

    updateDashboard();

    notify(`Loaded ${file.name} successfully.`);
  }

  catch (error) {
    notify(error.message || "Could not read this CSV file.");
  }
}

// UPDATE DASHBOARD
function updateDashboard() {
  const { headers, records, name } = dataset;

  $("stat-dataset").textContent =
    name.length > 19 ? name.slice(0, 16) + "…" : name;

  $("stat-dataset-sub").textContent = "CSV loaded in browser";

  $("stat-rows").textContent =
    records.length.toLocaleString();

  $("stat-cols").textContent = headers.length;

  $("dataset-name").textContent = name;
  $("dataset-status").textContent = "UPLOADED";

  $("dataset-empty").classList.add("hidden");
  $("dataset-details").classList.remove("hidden");

  $("detail-rows").textContent =
    records.length.toLocaleString();

  $("detail-cols").textContent = headers.length;

  const missing = records.reduce((sum, row) => {
    return sum + row.filter(value => value === "").length;
  }, 0);

  const duplicates =
    records.length -
    new Set(records.map(row => JSON.stringify(row))).size;

  $("detail-missing").textContent =
    missing.toLocaleString();

  $("detail-duplicates").textContent =
    duplicates.toLocaleString();

  $("analyze-btn").disabled = false;

  // DATASET PREVIEW
  const preview = $("preview-table");
  preview.innerHTML = "";

  const head = document.createElement("tr");

  headers.forEach(header => {
    const th = document.createElement("th");
    th.textContent = header;
    head.appendChild(th);
  });

  preview.appendChild(head);

  records.slice(0, 8).forEach(record => {
    const tr = document.createElement("tr");

    record.forEach(value => {
      const td = document.createElement("td");
      td.textContent = value || "—";
      tr.appendChild(td);
    });

    preview.appendChild(tr);
  });

  // TARGET COLUMN
  const target = $("target-column");

  target.disabled = false;
  target.innerHTML = "";

  headers.forEach(header => {
    const option = document.createElement("option");

    option.value = header;
    option.textContent = header;

    target.appendChild(option);
  });

  runAnalysis();
}

// BASIC DATA ANALYSIS
function runAnalysis() {
  if (!dataset) return;

  const { headers, records } = dataset;

  let numeric = 0;
  let categorical = 0;

  const profile = [];

  headers.forEach((header, index) => {
    const values = records
      .map(row => row[index])
      .filter(value => value !== "");

    const numericValues = values.filter(value => {
      return Number.isFinite(Number(value));
    });

    const isNumeric =
      values.length > 0 &&
      numericValues.length / values.length >= 0.85;

    if (isNumeric) {
      numeric++;
    } else {
      categorical++;
    }

    const unique = new Set(values).size;

    profile.push({
      header,
      type: isNumeric ? "Numeric" : "Categorical",
      missing: records.length - values.length,
      unique
    });
  });

  const missing = records.reduce((sum, row) => {
    return sum + row.filter(value => value === "").length;
  }, 0);

  const duplicates =
    records.length -
    new Set(records.map(row => JSON.stringify(row))).size;

  $("numeric-count").textContent = numeric;
  $("categorical-count").textContent = categorical;
  $("missing-count").textContent = missing.toLocaleString();
  $("duplicate-count").textContent = duplicates.toLocaleString();

  $("profile-count").textContent =
    headers.length + " columns";

  // COLUMN PROFILE TABLE
  const table = $("profile-table");

  table.innerHTML = `
    <tr>
      <th>Column</th>
      <th>Detected type</th>
      <th>Missing</th>
      <th>Unique values</th>
    </tr>
  `;

  profile.forEach(item => {
    const tr = document.createElement("tr");

    [
      item.header,
      item.type,
      item.missing,
      item.unique
    ].forEach(value => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });

    table.appendChild(tr);
  });
}

// ANALYZE BUTTON
$("analyze-btn").addEventListener("click", () => {
  runAnalysis();
  notify("Dataset profile updated.");
});
