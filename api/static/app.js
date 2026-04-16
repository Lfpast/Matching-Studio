const professorForm = document.getElementById("professorSearchForm");
const professorResultsList = document.getElementById("professorResultsList");
const professorResultsHint = document.getElementById("professorResultsHint");
const professorStatusLabel = document.getElementById("professorApiStatus");
const startupStatusLabel = document.getElementById("startupApiStatus");
const professorEmptyState = document.getElementById("professorEmptyState");
const professorPage = document.getElementById("professor-page");
const startupPage = document.getElementById("startup-page");
let currentDeepTechModal = null;
const SOURCE_ORDER = { bmh: 0, eas: 1, mes: 2 };
const HIGHLIGHT_STOPWORDS = new Set([
    "a", "an", "the", "and", "or", "but", "nor", "yet", "so",
    "for", "from", "with", "without", "into", "onto", "to", "of", "in", "on", "at", "by", "as",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "can", "could", "should",
    "would", "will", "may", "might", "must", "this", "that", "these", "those", "it", "its", "their",
    "there", "here", "what", "which", "when", "where", "who", "whom", "how", "why", "if", "then", "than",
]);
const CJK_HIGHLIGHT_STOPWORDS = new Set(["和", "及", "与", "以及", "并且", "或者"]);

function escapeHtml(text) {
    return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function getKeywords(query) {
    const latinTokens = query
        .toLowerCase()
        .split(/[^a-z0-9]+/g)
        .filter((token) => token.length >= 3 && !HIGHLIGHT_STOPWORDS.has(token));

    const cjkMatches = query.match(/[\u4e00-\u9fff]+/g) || [];
    const cjkTokens = cjkMatches.filter((token) => token.length >= 2 && !CJK_HIGHLIGHT_STOPWORDS.has(token));

    return [...new Set([...latinTokens, ...cjkTokens])];
}

function sanitizeKeywordList(keywords) {
    if (!Array.isArray(keywords)) return [];

    const seen = new Set();
    const sanitized = [];
    keywords.forEach((raw) => {
        const text = String(raw || "").trim();
        if (!text) return;

        const normalized = text.toLowerCase();
        if (HIGHLIGHT_STOPWORDS.has(normalized) || CJK_HIGHLIGHT_STOPWORDS.has(normalized)) {
            return;
        }

        const isCjk = /[\u4e00-\u9fff]/.test(text);
        if (!isCjk && normalized.length < 3) {
            return;
        }

        if (seen.has(normalized)) {
            return;
        }

        seen.add(normalized);
        sanitized.push(text);
    });

    return sanitized;
}

function highlightText(text, keywords) {
    let safe = escapeHtml(text);
    keywords.forEach((keyword) => {
        const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const flags = /[a-z0-9]/i.test(keyword) ? "gi" : "g";
        const regex = new RegExp(escaped, flags);
        safe = safe.replace(regex, (match) => `<mark>${match}</mark>`);
    });
    return safe;
}

/**
 * Produce a keyword-aware summary. Tries to pick fragments that contain
 * query keywords so the most relevant parts surface first.
 */
function smartSummarize(text, keywords, maxLength = 200) {
    if (!text) return "No research interests provided.";
    if (text.length <= maxLength) return text;

    // Split by comma / semicolon to get interest fragments
    const fragments = text.split(/[;,]/).map((s) => s.trim()).filter(Boolean);

    if (keywords.length === 0 || fragments.length <= 1) {
        return text.slice(0, maxLength).trim() + "…";
    }

    // Score each fragment by number of keyword matches
    const scored = fragments.map((frag) => {
        const lower = frag.toLowerCase();
        let score = 0;
        keywords.forEach((kw) => {
            if (lower.includes(kw.toLowerCase())) score += 1;
        });
        return { frag, score };
    });

    // Sort matched first, keep original order among equal scores
    scored.sort((a, b) => b.score - a.score);

    // Take fragments until we hit maxLength
    let result = "";
    for (const { frag } of scored) {
        const next = result ? result + ", " + frag : frag;
        if (next.length > maxLength) break;
        result = next;
    }

    if (!result) result = scored[0].frag.slice(0, maxLength);
    if (result.length < text.length) result += "…";
    return result;
}

/**
 * Extract matching keywords from research_interests that overlap with query.
 * Returns { matched: [...], context: [...] }
 */
function extractKeywordChips(interestsText, queryKeywords) {
    if (!interestsText) return { matched: [], context: [] };

    // Split interests into tokens/phrases
    const interestParts = interestsText
        .split(/[;,]/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0 && s.length < 60);

    const matched = [];
    const context = [];

    interestParts.forEach((part) => {
        const lower = part.toLowerCase();
        const isMatch = queryKeywords.some((kw) => lower.includes(kw.toLowerCase()));
        if (isMatch) {
            matched.push(part);
        } else if (context.length < 3) {
            context.push(part);
        }
    });

    return {
        matched: matched.slice(0, 5),
        context: context.slice(0, 3),
    };
}

function showProfessorEmptyState() {
    if (professorEmptyState) professorEmptyState.classList.remove("hidden");
}

function hideProfessorEmptyState() {
    if (professorEmptyState) professorEmptyState.classList.add("hidden");
}

function unlockProfessorResults() {
    if (!professorPage || !professorPage.classList.contains("search-locked")) return;
    professorPage.classList.remove("search-locked");
    professorPage.classList.add("search-unlocked");
}

function unlockStartupResults() {
    if (!startupPage || !startupPage.classList.contains("search-locked")) return;
    startupPage.classList.remove("search-locked");
    startupPage.classList.add("search-unlocked");
}

function createTagList(items, className) {
    if (!items || items.length === 0) return "";
    return items
        .map((item) => `<span class="${className}">${escapeHtml(item)}</span>`)
        .join("");
}

function parseTechEdges(techEdgesText) {
    if (!techEdgesText) return [];

    const raw = String(techEdgesText).replace(/\r/g, "\n");
    const bulletSplit = raw
        .split(/\n+/)
        .flatMap((line) => line.split(/(?=•)/g))
        .map((segment) => segment.trim())
        .filter(Boolean);

    const normalized = bulletSplit.map((segment) => segment.replace(/^•\s*/, "").trim()).filter(Boolean);
    if (normalized.length > 0) {
        return normalized;
    }

    return raw
        .split(/[;|]+/)
        .map((segment) => segment.trim())
        .filter(Boolean);
}

function normalizeSource(source) {
    const value = String(source || "EAS").trim().toLowerCase();
    if (value === "bmh") return "bmh";
    if (value === "mes") return "mes";
    return "eas";
}

function sortProjectsBySource(projects) {
    return [...projects].sort((left, right) => {
        const leftSource = normalizeSource(left.source);
        const rightSource = normalizeSource(right.source);
        const sourceDiff = (SOURCE_ORDER[leftSource] ?? 99) - (SOURCE_ORDER[rightSource] ?? 99);
        if (sourceDiff !== 0) return sourceDiff;

        const leftTitle = String(left.technology_title || "");
        const rightTitle = String(right.technology_title || "");
        return leftTitle.localeCompare(rightTitle);
    });
}

function dominantSource(projects) {
    const counter = { eas: 0, bmh: 0, mes: 0 };
    projects.forEach((project) => {
        counter[normalizeSource(project.source)] += 1;
    });
    return Object.entries(counter).sort((a, b) => b[1] - a[1])[0][0] || "eas";
}

function closeDeepTechModal() {
    if (!currentDeepTechModal) return;

    const { overlay, modal, originRect } = currentDeepTechModal;
    overlay.classList.remove("visible");
    modal.classList.remove("animating-in");
    modal.classList.add("animating-out");

    modal.style.left = `${originRect.left}px`;
    modal.style.top = `${originRect.top}px`;
    modal.style.width = `${originRect.width}px`;
    modal.style.height = `${originRect.height}px`;

    const cleanup = () => {
        overlay.remove();
        modal.remove();
        currentDeepTechModal = null;
    };

    modal.addEventListener("transitionend", cleanup, { once: true });
    setTimeout(cleanup, 460);
}

function openDeepTechModal(professorName, projects, cardElement, fixedSource = null) {
    if (!projects || projects.length === 0 || !cardElement) return;
    if (currentDeepTechModal) {
        closeDeepTechModal();
    }

    const originRect = cardElement.getBoundingClientRect();
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";

    const modal = document.createElement("div");
    const modalSource = fixedSource ? normalizeSource(fixedSource) : dominantSource(projects);
    modal.className = `deeptech-modal deeptech-modal-${modalSource}`;
    modal.style.left = `${originRect.left}px`;
    modal.style.top = `${originRect.top}px`;
    modal.style.width = `${originRect.width}px`;
    modal.style.height = `${originRect.height}px`;

    const projectCards = sortProjectsBySource(projects)
        .map((project) => {
            const source = normalizeSource(project.source);
            const edgeItems = parseTechEdges(project.tech_edges || "")
                .slice(0, 20)
                .map((edge) => `<li class="deeptech-edge-item">${escapeHtml(edge)}</li>`)
                .join("");

            return `
      <article class="deeptech-project-card deeptech-project-card-${source}">
        <div class="deeptech-project-header">
          <h4>${escapeHtml(project.technology_title || "Untitled Technology")}</h4>
          ${project.cluster ? `<span class="cluster-chip cluster-chip-${source}">${escapeHtml(project.cluster)}</span>` : ""}
        </div>
        <div class="deeptech-meta-row">
          ${project.trl ? `<span class="deeptech-meta-tag">TRL: ${escapeHtml(project.trl)}</span>` : ""}
          ${project.ip_status ? `<span class="deeptech-meta-tag">IP: ${escapeHtml(project.ip_status)}</span>` : ""}
          <span class="deeptech-meta-tag">Relevance: ${(project.relevance_score || 0).toFixed(3)}</span>
        </div>
        <div class="deeptech-overview">${escapeHtml(project.overview || "No overview provided.")}</div>
        ${edgeItems ? `<div class="deeptech-list-row"><strong>Key Technology Edges</strong><ul class="deeptech-edge-list">${edgeItems}</ul></div>` : ""}
        ${project.applications?.length ? `<div class="deeptech-list-row"><strong>Potential Applications</strong><div>${createTagList(project.applications, "deeptech-app-tag")}</div></div>` : ""}
        ${project.industries?.length ? `<div class="deeptech-list-row"><strong>Applicable Industry</strong><div>${createTagList(project.industries, "deeptech-industry-tag")}</div></div>` : ""}
      </article>`;
        })
        .join("");

    modal.innerHTML = `
    <button class="modal-close-btn" type="button" aria-label="Close">✕</button>
    <div class="deeptech-modal-header">
      <h3>${escapeHtml(professorName)} — DeepTech Projects</h3>
    </div>
    <div class="deeptech-modal-content">${projectCards}</div>
  `;

    document.body.appendChild(overlay);
    document.body.appendChild(modal);
    currentDeepTechModal = { overlay, modal, originRect };

    const closeBtn = modal.querySelector(".modal-close-btn");
    closeBtn.addEventListener("click", closeDeepTechModal);
    overlay.addEventListener("click", closeDeepTechModal);

    requestAnimationFrame(() => {
        const targetWidth = Math.min(700, window.innerWidth - 40);
        const targetHeight = Math.min(window.innerHeight * 0.92, Math.max(420, window.innerHeight * 0.75));
        const targetLeft = (window.innerWidth - targetWidth) / 2;
        const targetTop = (window.innerHeight - targetHeight) / 2;

        overlay.classList.add("visible");
        modal.classList.add("animating-in");
        modal.style.left = `${targetLeft}px`;
        modal.style.top = `${targetTop}px`;
        modal.style.width = `${targetWidth}px`;
        modal.style.height = `${targetHeight}px`;
    });
}

async function checkHealth() {
    try {
        const response = await fetch("/health");
        if (!response.ok) {
            throw new Error("health failed");
        }
        professorStatusLabel.textContent = "API online";
        if (startupStatusLabel) startupStatusLabel.textContent = "API online";
    } catch (error) {
        professorStatusLabel.textContent = "API offline";
        if (startupStatusLabel) startupStatusLabel.textContent = "API offline";
    }
}

function renderProfessorResults(results, query, extractedKeywords = []) {
    professorResultsList.innerHTML = "";
    if (!results || results.length === 0) {
        professorResultsHint.textContent = "No matches yet. Try another query.";
        showProfessorEmptyState();
        return;
    }

    hideProfessorEmptyState();
    professorResultsHint.textContent = `Showing ${results.length} matches`;
    const backendKeywords = sanitizeKeywordList(extractedKeywords);
    const keywords = backendKeywords.length > 0 ? backendKeywords : getKeywords(query || "");

    results.forEach((item, index) => {
        const card = document.createElement("article");
        card.className = "result-card";

        const fullInterests = item.research_interests || "";
        const summary = smartSummarize(fullInterests, keywords);
        const highlightedSummary = highlightText(summary, keywords);

        // Keyword chips
        const chips = extractKeywordChips(fullInterests, keywords);
        let chipsHtml = "";
        if (chips.matched.length > 0 || chips.context.length > 0) {
            chipsHtml = '<div class="keyword-chips">';
            chips.matched.forEach((kw) => {
                chipsHtml += `<span class="keyword-chip matched">${escapeHtml(kw)}</span>`;
            });
            chips.context.forEach((kw) => {
                chipsHtml += `<span class="keyword-chip context">${escapeHtml(kw)}</span>`;
            });
            chipsHtml += "</div>";
        }

        const deeptechProjects = sortProjectsBySource(item.deeptech_projects || []);
        let deeptechHtml = "";
        if (deeptechProjects.length > 0) {
            const deeptechListClass = "keyword-chips deeptech-chip-list has-scroll";
            const chips = deeptechProjects
                .map((project) => {
                    const source = normalizeSource(project.source);
                    return `<span class="deeptech-chip deeptech-chip-${source} deeptech-chip-trigger" role="button" tabindex="0" data-source="${source}">${escapeHtml(project.technology_title || "Untitled")}</span>`;
                })
                .join("");
            deeptechHtml = `
        <section class="deeptech-section">
          <p><strong>DeepTech Projects</strong></p>
          <div class="${deeptechListClass}" data-lenis-prevent>${chips}</div>
        </section>
      `;
        }

        // Full interests (expandable)
        const showToggle = fullInterests.length > 200;
        const fullHighlighted = highlightText(fullInterests, keywords);
        const hasUrl = typeof item.url === "string" && item.url.trim().length > 0;
        const professorNameHtml = hasUrl
            ? `<a class="prof-name-link" href="${escapeHtml(item.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.name)}</a>`
            : `${escapeHtml(item.name)}`;

        card.innerHTML = `
      <span class="badge">#${index + 1} Ranked</span>
      <h3>${professorNameHtml}</h3>
      <p>${escapeHtml(item.department)}</p>
      <p>${escapeHtml(item.title)}</p>
      <p class="interest-summary"><strong>Interests:</strong> ${highlightedSummary}</p>
      ${showToggle ? `<div class="interest-full">${fullHighlighted}</div><button class="toggle-interests" type="button">Show all ▼</button>` : ""}
      ${chipsHtml}
      ${deeptechHtml}
      <div class="kpi">
        <span>Score: ${item.score.toFixed(3)}</span>
        <span>Sim: ${item.similarity.toFixed(3)}</span>
        <span>Priority: ${item.priority_score.toFixed(3)}</span>
      </div>
    `;

        // Toggle expand/collapse
        if (showToggle) {
            const toggleBtn = card.querySelector(".toggle-interests");
            const fullDiv = card.querySelector(".interest-full");
            toggleBtn.addEventListener("click", () => {
                const isExpanded = fullDiv.classList.toggle("expanded");
                toggleBtn.textContent = isExpanded ? "Collapse ▲" : "Show all ▼";
            });
            toggleBtn.addEventListener("click", (evt) => evt.stopPropagation());
        }

        const nameLink = card.querySelector(".prof-name-link");
        if (nameLink) {
            nameLink.addEventListener("click", (evt) => evt.stopPropagation());
        }

        const sourceTriggers = card.querySelectorAll(".deeptech-chip-trigger");
        sourceTriggers.forEach((trigger) => {
            const handleOpenBySource = () => {
                const source = normalizeSource(trigger.dataset.source);
                const sourceProjects = deeptechProjects.filter(
                    (project) => normalizeSource(project.source) === source
                );
                openDeepTechModal(item.name, sourceProjects, card, source);
            };

            trigger.addEventListener("click", (evt) => {
                evt.stopPropagation();
                handleOpenBySource();
            });

            trigger.addEventListener("keydown", (evt) => {
                if (evt.key === "Enter" || evt.key === " ") {
                    evt.preventDefault();
                    evt.stopPropagation();
                    handleOpenBySource();
                }
            });

            trigger.addEventListener("mousemove", (evt) => {
                const rect = trigger.getBoundingClientRect();
                const x = ((evt.clientX - rect.left) / rect.width) * 100;
                const y = ((evt.clientY - rect.top) / rect.height) * 100;
                trigger.style.setProperty("--mx", `${x}%`);
                trigger.style.setProperty("--my", `${y}%`);
            });

            trigger.addEventListener("mouseleave", () => {
                trigger.style.setProperty("--mx", "50%");
                trigger.style.setProperty("--my", "50%");
            });
        });

        professorResultsList.appendChild(card);
    });
}

professorForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = document.getElementById("professorQuery").value.trim();

    if (!query) {
        professorResultsHint.textContent = "Please enter a query.";
        return;
    }

    professorResultsHint.textContent = "Running matching...";

    const payload = {
        query,
        top_k: Number(document.getElementById("professorTopK").value),
        alpha: Number(document.getElementById("professorAlpha").value),
        beta: Number(document.getElementById("professorBeta").value),
        graph_neighbor_weight: Number(document.getElementById("professorNeighborWeight").value),
        mode: "professor",
    };

    try {
        const response = await fetch("/match", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error("match failed");
        }

        const data = await response.json();
        unlockProfessorResults();
        const extractedKeywords = Array.isArray(data.keywords)
            ? data.keywords.map((item) => (typeof item === "string" ? item : item.keyword))
            : [];
        renderProfessorResults(data.results || [], query, extractedKeywords);
    } catch (error) {
        professorResultsHint.textContent = "Failed to fetch results. Check the API.";
    }
});

checkHealth();

class StartupSearchManager {
    constructor() {
        this.currentStartupModal = null;
        this.initElements();
        this.bindEvents();
    }

    initElements() {
        this.form = document.getElementById("startupForm");
        this.queryInput = document.getElementById("startupQuery");
        this.resultsList = document.getElementById("startupResultsList");
        this.resultsHint = document.getElementById("startupResultsHint");
        this.emptyState = document.getElementById("startupEmptyState");
    }

    bindEvents() {
        if (!this.form) return;
        this.form.addEventListener("submit", async (event) => {
            await this.submitStartupMatch(event);
        });
    }

    collectParams() {
        return {
            query: this.queryInput ? this.queryInput.value.trim() : "",
            top_k: Number(document.getElementById("startupTopK")?.value || 5),
            alpha: 1.0,
            beta: 0.0,
            graph_neighbor_weight: Number(document.getElementById("startupNeighborWeight")?.value || 0.15),
            mode: "startup",
        };
    }

    async submitStartupMatch(event) {
        event.preventDefault();

        const payload = this.collectParams();
        if (!payload.query) {
            if (this.resultsHint) this.resultsHint.textContent = "Please enter a query.";
            return;
        }

        if (this.resultsHint) this.resultsHint.textContent = "Running matching...";

        try {
            const response = await fetch("/match", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                throw new Error("startup match failed");
            }

            const data = await response.json();
            unlockStartupResults();
            const extractedKeywords = Array.isArray(data.keywords)
                ? data.keywords.map((item) => (typeof item === "string" ? item : item.keyword))
                : [];

            this.renderStartupResults(data.startup_results || [], extractedKeywords);
            if (data.status === "invalid" && this.resultsHint) {
                this.resultsHint.textContent = data.message || "Query is invalid.";
            }
        } catch (error) {
            if (this.resultsHint) this.resultsHint.textContent = "Failed to fetch startup results. Check the API.";
        }
    }

    showEmptyState() {
        if (this.emptyState) this.emptyState.classList.remove("hidden");
    }

    hideEmptyState() {
        if (this.emptyState) this.emptyState.classList.add("hidden");
    }

    renderStartupResults(items, extractedKeywords = []) {
        if (!this.resultsList) return;

        this.resultsList.innerHTML = "";
        if (!items || items.length === 0) {
            if (this.resultsHint) this.resultsHint.textContent = "No startup matches yet. Try another query.";
            this.showEmptyState();
            return;
        }

        this.hideEmptyState();
        if (this.resultsHint) this.resultsHint.textContent = `Showing ${items.length} startup matches`;

        const backendKeywords = sanitizeKeywordList(extractedKeywords);
        const keywords = backendKeywords.length > 0 ? backendKeywords : getKeywords(this.queryInput?.value || "");

        items.forEach((item, index) => {
            const card = this.renderStartupCard(item, keywords, index);
            this.resultsList.appendChild(card);
        });
    }

    renderStartupCard(item, keywords, index) {
        const card = document.createElement("article");
        card.className = "result-card startup-result-card";

        const hasWebsite = typeof item.website === "string" && item.website.trim().length > 0;
        const companyName = String(item.company_name || "Unnamed Startup");
        const companyHtml = hasWebsite
            ? `<a class="startup-name-link" href="${escapeHtml(item.website)}" target="_blank" rel="noopener noreferrer">${escapeHtml(companyName)}</a>`
            : escapeHtml(companyName);

        const people = Array.isArray(item.people) ? item.people : [];
        const categories = Array.isArray(item.categories) ? item.categories : [];
        const matchedKeywords = Array.isArray(item.matched_keywords) ? item.matched_keywords : [];
        const highlightKeywords = sanitizeKeywordList(matchedKeywords.length > 0 ? matchedKeywords : keywords);
        const sourceYearText = item.source_year ? String(item.source_year) : "";

        const peopleText = people.length > 0 ? people.join("; ") : "";
        const refCode = String(item.ref_code || "").trim();
        const descriptionText = String(item.description || "").trim();

        const categoryHtml = categories.length > 0
            ? categories.map((category) => `<span class="startup-category-chip">${escapeHtml(String(category))}</span>`).join("")
            : "";

        const peopleRow = peopleText
            ? `<div class="startup-meta-row"><strong>People:</strong> ${escapeHtml(peopleText)}</div>`
            : "";

        const refCodeRow = refCode
            ? `<div class="startup-meta-row"><strong>Ref. Code:</strong> ${escapeHtml(refCode)}</div>`
            : "";

        const sourceYearRow = sourceYearText
            ? `<div class="startup-meta-row"><strong>Source Year:</strong> ${escapeHtml(sourceYearText)}</div>`
            : "";

        const descriptionRow = descriptionText
            ? `<p class="startup-description"><strong>Description:</strong> ${highlightText(descriptionText, highlightKeywords)}</p>`
            : "";

        card.innerHTML = `
      <span class="badge">#${index + 1} Ranked</span>
      <h3>${companyHtml}</h3>
      ${peopleRow}
      ${refCodeRow}
      ${sourceYearRow}
      ${descriptionRow}
      ${categoryHtml ? `<div class="startup-category-list">${categoryHtml}</div>` : ""}
      <div class="kpi startup-kpi"><span>Score: ${Number(item.score || 0).toFixed(3)}</span></div>
    `;

        const titleLink = card.querySelector(".startup-name-link");
        if (titleLink) {
            titleLink.addEventListener("click", (evt) => evt.stopPropagation());
        }

        card.addEventListener("click", () => {
            this.openStartupDetailModal(item, card, highlightKeywords);
        });

        return card;
    }

    closeStartupDetailModal() {
        if (!this.currentStartupModal) return;

        const { overlay, modal, originRect } = this.currentStartupModal;
        overlay.classList.remove("visible");
        modal.classList.remove("animating-in");
        modal.classList.add("animating-out");

        modal.style.left = `${originRect.left}px`;
        modal.style.top = `${originRect.top}px`;
        modal.style.width = `${originRect.width}px`;
        modal.style.height = `${originRect.height}px`;

        const cleanup = () => {
            overlay.remove();
            modal.remove();
            this.currentStartupModal = null;
        };

        modal.addEventListener("transitionend", cleanup, { once: true });
        setTimeout(cleanup, 460);
    }

    openStartupDetailModal(item, cardElement, keywords) {
        if (!item || !cardElement) return;
        if (this.currentStartupModal) {
            this.closeStartupDetailModal();
        }

        const originRect = cardElement.getBoundingClientRect();
        const overlay = document.createElement("div");
        overlay.className = "modal-overlay";

        const modal = document.createElement("div");
        modal.className = "startup-modal";
        modal.style.left = `${originRect.left}px`;
        modal.style.top = `${originRect.top}px`;
        modal.style.width = `${originRect.width}px`;
        modal.style.height = `${originRect.height}px`;

        const people = Array.isArray(item.people) ? item.people : [];
        const categories = Array.isArray(item.categories) ? item.categories : [];
        const tels = Array.isArray(item.tels) ? item.tels : [];
        const emails = Array.isArray(item.emails) ? item.emails : [];

        const sectionRows = [];
        if (item.description) {
            sectionRows.push(`
        <section class="startup-detail-section">
          <h4>Description</h4>
          <p class="startup-detail-text">${highlightText(String(item.description), keywords)}</p>
        </section>
      `);
        }

        if (categories.length > 0) {
            sectionRows.push(`
        <section class="startup-detail-section">
          <h4>Categories</h4>
          <div class="startup-category-list">
            ${categories.map((category) => `<span class="startup-category-chip">${escapeHtml(String(category))}</span>`).join("")}
          </div>
        </section>
      `);
        }

        const infoGridRows = [];
        if (item.source_year) infoGridRows.push(`<div class="startup-info-item"><span class="label">Source Year</span><span class="value">${escapeHtml(String(item.source_year))}</span></div>`);
        if (item.ref_code) infoGridRows.push(`<div class="startup-info-item"><span class="label">Ref. Code</span><span class="value">${escapeHtml(String(item.ref_code))}</span></div>`);
        if (item.funding) infoGridRows.push(`<div class="startup-info-item"><span class="label">Funding</span><span class="value">${escapeHtml(String(item.funding))}</span></div>`);
        if (item.background_year) infoGridRows.push(`<div class="startup-info-item"><span class="label">Background (Year)</span><span class="value">${escapeHtml(String(item.background_year))}</span></div>`);
        if (people.length > 0) infoGridRows.push(`<div class="startup-info-item"><span class="label">People</span><span class="value">${escapeHtml(people.join("; "))}</span></div>`);

        if (infoGridRows.length > 0) {
            sectionRows.push(`
        <section class="startup-detail-section">
          <h4>Details</h4>
          <div class="startup-info-grid">${infoGridRows.join("")}</div>
        </section>
      `);
        }

        if (tels.length > 0) {
            sectionRows.push(`
        <section class="startup-detail-section">
          <h4>Tel</h4>
          <div class="startup-contact-list">
            ${tels.map((tel) => `
              <div class="startup-contact-item">
                <span>${escapeHtml(String(tel))}</span>
                <button class="contact-copy-btn" type="button" data-copy="${escapeHtml(String(tel))}" data-type="tel" data-startup-id="${escapeHtml(String(item.startup_id || ""))}">Copy</button>
              </div>
            `).join("")}
          </div>
        </section>
      `);
        }

        if (emails.length > 0) {
            sectionRows.push(`
        <section class="startup-detail-section">
          <h4>Email</h4>
          <div class="startup-contact-list">
            ${emails.map((email) => `
              <div class="startup-contact-item">
                <span>${escapeHtml(String(email))}</span>
                <button class="contact-copy-btn" type="button" data-copy="${escapeHtml(String(email))}" data-type="email" data-startup-id="${escapeHtml(String(item.startup_id || ""))}">Copy</button>
              </div>
            `).join("")}
          </div>
        </section>
      `);
        }

        const hasWebsite = typeof item.website === "string" && item.website.trim().length > 0;
        const titleHtml = hasWebsite
            ? `<a class="startup-modal-title-link" href="${escapeHtml(item.website)}" target="_blank" rel="noopener noreferrer">${escapeHtml(String(item.company_name || "Unnamed Startup"))}</a>`
            : `${escapeHtml(String(item.company_name || "Unnamed Startup"))}`;

        modal.innerHTML = `
      <button class="modal-close-btn" type="button" aria-label="Close">✕</button>
      <div class="startup-modal-header">
        <h3>${titleHtml}</h3>
      </div>
      <div class="startup-modal-content">
        ${sectionRows.join("")}
      </div>
    `;

        document.body.appendChild(overlay);
        document.body.appendChild(modal);
        this.currentStartupModal = { overlay, modal, originRect };

        const closeBtn = modal.querySelector(".modal-close-btn");
        closeBtn.addEventListener("click", () => this.closeStartupDetailModal());
        overlay.addEventListener("click", () => this.closeStartupDetailModal());

        const titleLink = modal.querySelector(".startup-modal-title-link");
        if (titleLink) {
            titleLink.addEventListener("click", (evt) => evt.stopPropagation());
        }

        const copyButtons = modal.querySelectorAll(".contact-copy-btn");
        copyButtons.forEach((button) => {
            button.addEventListener("click", async (evt) => {
                evt.stopPropagation();
                const copyText = button.dataset.copy || "";
                const type = button.dataset.type || "contact";
                const startupId = button.dataset.startupId || "";
                await this.copyContact(copyText, type, startupId, button);
            });
        });

        requestAnimationFrame(() => {
            const targetWidth = Math.min(760, window.innerWidth - 40);
            const targetHeight = Math.min(window.innerHeight * 0.92, Math.max(440, window.innerHeight * 0.75));
            const targetLeft = (window.innerWidth - targetWidth) / 2;
            const targetTop = (window.innerHeight - targetHeight) / 2;

            overlay.classList.add("visible");
            modal.classList.add("animating-in");
            modal.style.left = `${targetLeft}px`;
            modal.style.top = `${targetTop}px`;
            modal.style.width = `${targetWidth}px`;
            modal.style.height = `${targetHeight}px`;
        });
    }

    async copyContact(text, type, startupId, button) {
        const value = String(text || "").trim();
        if (!value) return;

        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(value);
            } else {
                const tempInput = document.createElement("textarea");
                tempInput.value = value;
                tempInput.setAttribute("readonly", "");
                tempInput.style.position = "fixed";
                tempInput.style.opacity = "0";
                document.body.appendChild(tempInput);
                tempInput.select();
                document.execCommand("copy");
                tempInput.remove();
            }
            if (button) {
                const previous = button.textContent;
                button.classList.add("copied");
                button.textContent = "Copied";
                setTimeout(() => {
                    button.classList.remove("copied");
                    button.textContent = previous || "Copy";
                }, 1200);
            }
        } catch (_error) {
            if (button) {
                const previous = button.textContent;
                button.textContent = "Failed";
                setTimeout(() => {
                    button.textContent = previous || "Copy";
                }, 1200);
            }
            console.error(`Failed to copy ${type} for startup ${startupId}`);
        }
    }
}

// --- Database Update Interface code added below ---
class AuthManager {
    constructor() {
        this.token = this.getStoredToken();
        this.username = this.getStoredUsername();
        this.rememberMe = this.getRememberMeSetting();
    }

    async login(username, password, rememberMe) {
        try {
            const response = await fetch('/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Login failed');
            }

            const data = await response.json();
            this.token = data.token;
            this.username = username;
            this.rememberMe = rememberMe;

            // 清理旧token，避免local/session冲突导致读取到过期token
            localStorage.removeItem('auth_token');
            localStorage.removeItem('auth_username');
            sessionStorage.removeItem('auth_token');
            sessionStorage.removeItem('auth_username');

            // 保存token和用户信息
            if (rememberMe) {
                localStorage.setItem('auth_token', this.token);
                localStorage.setItem('auth_username', username);
                localStorage.setItem('remember_me', 'true');
            } else {
                sessionStorage.setItem('auth_token', this.token);
                sessionStorage.setItem('auth_username', username);
            }

            return true;
        } catch (error) {
            console.error('Login error:', error);
            return false;
        }
    }

    logout() {
        this.token = null;
        this.username = null;
        localStorage.removeItem('auth_token');
        localStorage.removeItem('auth_username');
        localStorage.removeItem('remember_me');
        sessionStorage.removeItem('auth_token');
        sessionStorage.removeItem('auth_username');
    }

    getStoredToken() {
        return localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
    }

    getStoredUsername() {
        return localStorage.getItem('auth_username') || sessionStorage.getItem('auth_username');
    }

    syncTokenFromStorage() {
        const storedToken = this.getStoredToken();
        if (storedToken) {
            this.token = storedToken;
        }
        return this.token;
    }

    getRememberMeSetting() {
        return localStorage.getItem('remember_me') === 'true';
    }

    isAuthenticated() {
        return !!this.token;
    }

    getAuthHeader() {
        const token = this.syncTokenFromStorage();
        return token ? { 'Authorization': `Bearer ${token}` } : {};
    }
}

class SidebarManager {
    constructor(authManager) {
        this.authManager = authManager;
        this.hamburger = document.getElementById('professor-hamburger-btn');
        this.hamburgerStartup = document.getElementById('hamburger-btn-startup');
        this.hamburgerUpdate = document.getElementById('hamburger-btn-update'); // The other page
        this.sidebar = document.getElementById('sidebar');
        this.updateLink = document.querySelector('[data-page="update"]');
        this.startupLink = document.querySelector('[data-page="startup"]');
        this.professorLink = document.querySelector('[data-page="professor"]');
        this.logoutBtn = document.getElementById('logout-btn');

        this.initializeEventListeners();
        this.updateAuthUI();
    }

    updateAuthUI() {
        if (this.authManager.isAuthenticated()) {
            this.logoutBtn.style.display = 'block';
        } else {
            this.logoutBtn.style.display = 'none';
        }
    }

    initializeEventListeners() {
        // 汉堡菜单切换
        if (this.hamburger) {
            this.hamburger.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleSidebar();
            });
        }
        if (this.hamburgerUpdate) {
            this.hamburgerUpdate.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleSidebar();
            });
        }
        if (this.hamburgerStartup) {
            this.hamburgerStartup.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleSidebar();
            });
        }

        // 点击外部或页面其他区域收回侧边栏 (Bug 2, 5)
        document.addEventListener('click', (e) => {
            if (this.sidebar.classList.contains('open') &&
                !this.sidebar.contains(e.target)) {
                this.closeSidebar();
            }
        });

        // 防止点击侧边栏内部时关闭
        this.sidebar.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        // 导航链接
        if (this.updateLink) {
            this.updateLink.addEventListener('click', (e) => this.handleNavigationClick(e, 'update'));
        }
        if (this.professorLink) {
            this.professorLink.addEventListener('click', (e) => this.handleNavigationClick(e, 'professor'));
        }
        if (this.startupLink) {
            this.startupLink.addEventListener('click', (e) => this.handleNavigationClick(e, 'startup'));
        }

        // 登出按钮
        if (this.logoutBtn) {
            this.logoutBtn.addEventListener('click', () => this.handleLogout());
        }
    }

    handleNavigationClick(e, pageType) {
        e.preventDefault();

        // 如果访问update页面且未认证，显示认证弹窗
        if (pageType === 'update' && !this.authManager.isAuthenticated()) {
            showAuthModal(this.authManager);
            return;
        }

        this.navigateToPage(pageType);
        this.closeSidebar();
    }

    navigateToPage(pageType) {
        // 隐藏所有页面
        document.querySelectorAll('.page').forEach(page => {
            page.classList.remove('active');
        });

        // 显示指定页面
        const page = document.getElementById(`${pageType}-page`);
        if (page) {
            page.classList.add('active');
        }

        // 更新活跃导航项
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        const tgt = document.querySelector(`[data-page="${pageType}"]`);
        if (tgt) tgt.classList.add('active');
    }

    toggleSidebar() {
        this.sidebar.classList.toggle('open');
    }

    closeSidebar() {
        this.sidebar.classList.remove('open');
    }

    handleLogout() {
        this.authManager.logout();
        this.updateAuthUI();
        this.closeSidebar();
        this.navigateToPage('professor');
    }
}

function showAuthModal(authManager) {
    const modal = document.getElementById('auth-modal');
    const form = document.getElementById('auth-form');
    const errorMsg = document.getElementById('auth-error');
    const usernameInput = document.getElementById('username-input');
    const passwordInput = document.getElementById('password-input');
    const rememberMeCheckbox = document.getElementById('remember-me');

    // 清除之前的错误信息
    errorMsg.textContent = '';
    errorMsg.classList.remove('show');

    // 显示弹窗
    modal.classList.remove('hidden');

    // 焦点设置到用户名输入框
    usernameInput.focus();

    // 处理表单提交
    form.onsubmit = async (e) => {
        e.preventDefault();

        const username = usernameInput.value.trim();
        const password = passwordInput.value;
        const rememberMe = rememberMeCheckbox.checked;

        const success = await authManager.login(username, password, rememberMe);

        if (success) {
            // 关闭弹窗
            modal.classList.add('hidden');
            form.reset();

            // 重新导航到update页面
            const sidebarManager = window.sidebarManager; // 全局引用
            if (sidebarManager) {
                sidebarManager.updateAuthUI();
                sidebarManager.navigateToPage('update');
                sidebarManager.closeSidebar(); // 确保成功登录后侧边栏收回 (Bug 4)
            }
        } else {
            // 显示错误信息
            errorMsg.textContent = 'Invalid username or password';
            errorMsg.classList.add('show');
            passwordInput.value = '';
            passwordInput.focus();
        }
    };
}

// 点击弹窗外关闭
const authModalElement = document.getElementById('auth-modal');
if (authModalElement) {
    authModalElement.addEventListener('click', (e) => {
        if (e.target.id === 'auth-modal') {
            document.getElementById('auth-modal').classList.add('hidden');
        }
    });
}

class DatabaseUpdateManager {
    constructor(authManager) {
        this.authManager = authManager;
        this.inputCsvFile = null;
        this.deeptechXlsxFile = null;
        this.startupXlsxFile = null;
        this.isCsvUploaded = false; // Add state to track successful upload
        this.wsSocket = null;
        this.taskId = null;
        this.isUpdating = false;
        this.boundSyncUploadWindowHeights = () => this.syncUploadWindowHeights();
        this.boundSyncMonitorPanelHeight = () => this.syncMonitorPanelHeight();
        window.addEventListener('resize', this.boundSyncUploadWindowHeights);
        window.addEventListener('resize', this.boundSyncMonitorPanelHeight);

        this.initializeUI();
        this.bindEventListeners();
    }

    initializeUI() {
        // 文件选择按钮绑定
        const inputCsvBtn = document.getElementById('input-csv-btn');
        if (inputCsvBtn) {
            inputCsvBtn.addEventListener('click', () => {
                document.getElementById('input-csv-file').click();
            });
        }

        const dtxlsxBtn = document.getElementById('deeptech-xlsx-btn');
        if (dtxlsxBtn) {
            dtxlsxBtn.addEventListener('click', () => {
                document.getElementById('deeptech-xlsx-file').click();
            });
        }

        const startupXlsxBtn = document.getElementById('startup-xlsx-btn');
        if (startupXlsxBtn) {
            startupXlsxBtn.addEventListener('click', () => {
                document.getElementById('startup-xlsx-file').click();
            });
        }

        // 文件选择事件
        const inputCsvFileEl = document.getElementById('input-csv-file');
        if (inputCsvFileEl) {
            inputCsvFileEl.addEventListener('change', (e) => {
                this.handleCsvFileSelect(e);
            });
        }

        const dtxlsxFileEl = document.getElementById('deeptech-xlsx-file');
        if (dtxlsxFileEl) {
            dtxlsxFileEl.addEventListener('change', (e) => {
                this.handleXlsxFileSelect(e);
            });
        }

        const startupXlsxFileEl = document.getElementById('startup-xlsx-file');
        if (startupXlsxFileEl) {
            startupXlsxFileEl.addEventListener('change', (e) => {
                this.handleStartupXlsxFileSelect(e);
            });
        }

        this.initializeDropZones();
        requestAnimationFrame(() => this.syncUploadWindowHeights());
    }

    initializeDropZones() {
        const dropZoneConfigs = {
            'input-csv': {
                endpoint: '/api/upload/input-csv',
                inputId: 'input-csv-file',
                accept: ['.csv']
            },
            'deeptech-xlsx': {
                endpoint: '/api/upload/deeptech-xlsx',
                inputId: 'deeptech-xlsx-file',
                accept: ['.xlsx', '.xls']
            },
            'startup-xlsx': {
                endpoint: '/api/upload/startup-xlsx',
                inputId: 'startup-xlsx-file',
                accept: ['.xlsx', '.xls']
            }
        };

        document.querySelectorAll('.upload-window[data-upload-type]').forEach((zone) => {
            const fileType = zone.dataset.uploadType;
            const config = dropZoneConfigs[fileType];
            if (!config) {
                return;
            }

            let dragDepth = 0;

            const clearDragState = () => {
                dragDepth = 0;
                zone.classList.remove('drag-over');
            };

            zone.addEventListener('click', (e) => {
                if (e.target.closest('button')) {
                    return;
                }
                const inputEl = document.getElementById(config.inputId);
                if (inputEl) {
                    inputEl.click();
                }
            });

            zone.addEventListener('dragenter', (e) => {
                e.preventDefault();
                dragDepth += 1;
                zone.classList.add('drag-over');
            });

            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                if (e.dataTransfer) {
                    e.dataTransfer.dropEffect = 'copy';
                }
                zone.classList.add('drag-over');
            });

            zone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                dragDepth = Math.max(0, dragDepth - 1);
                if (dragDepth === 0 && !zone.contains(e.relatedTarget)) {
                    zone.classList.remove('drag-over');
                }
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                clearDragState();

                const droppedFile = e.dataTransfer && e.dataTransfer.files ? e.dataTransfer.files[0] : null;
                if (!droppedFile) {
                    return;
                }

                if (!this.isAcceptedUploadFile(droppedFile, config.accept)) {
                    this.showMessage(`Please drop a valid ${fileType === 'input-csv' ? 'CSV' : 'XLSX'} file.`, 'error');
                    return;
                }

                if (fileType === 'input-csv') {
                    this.handleCsvFileSelect({ target: { files: [droppedFile] } });
                } else if (fileType === 'deeptech-xlsx') {
                    this.handleXlsxFileSelect({ target: { files: [droppedFile] } });
                } else {
                    this.handleStartupXlsxFileSelect({ target: { files: [droppedFile] } });
                }
            });
        });
    }

    bindEventListeners() {
        const subCsvBtn = document.getElementById('submit-csv-btn');
        if (subCsvBtn) {
            subCsvBtn.addEventListener('click', () => {
                if (this.inputCsvFile) {
                    this.uploadFile(this.inputCsvFile, '/api/upload/input-csv', 'input-csv');
                }
            });
        }

        const subXlsxBtn = document.getElementById('submit-xlsx-btn');
        if (subXlsxBtn) {
            subXlsxBtn.addEventListener('click', () => {
                if (this.deeptechXlsxFile) {
                    this.uploadFile(this.deeptechXlsxFile, '/api/upload/deeptech-xlsx', 'deeptech-xlsx');
                }
            });
        }

        const subStartupBtn = document.getElementById('submit-startup-btn');
        if (subStartupBtn) {
            subStartupBtn.addEventListener('click', () => {
                if (this.startupXlsxFile) {
                    this.uploadFile(this.startupXlsxFile, '/api/upload/startup-xlsx', 'startup-xlsx');
                }
            });
        }

        const startBtn = document.getElementById('start-update-btn');
        if (startBtn) {
            startBtn.addEventListener('click', () => {
                this.startUpdate();
            });
        }

        const dlBtn = document.getElementById('download-report-btn');
        if (dlBtn) {
            dlBtn.addEventListener('click', () => {
                this.downloadReport();
            });
        }
    }

    handleCsvFileSelect(e) {
        this.inputCsvFile = e.target.files[0];
        if (this.inputCsvFile) {
            const chooseBtn = document.getElementById('input-csv-btn');
            chooseBtn.textContent = 'Choose File';
            chooseBtn.title = this.inputCsvFile.name;
            document.getElementById('input-csv-status').textContent = `✓ Selected: ${this._truncateFileName(this.inputCsvFile.name)}`;
        }
        this.updateStartButtonState();
        requestAnimationFrame(() => this.syncUploadWindowHeights());
    }

    handleXlsxFileSelect(e) {
        this.deeptechXlsxFile = e.target.files[0];
        if (this.deeptechXlsxFile) {
            const chooseBtn = document.getElementById('deeptech-xlsx-btn');
            chooseBtn.textContent = 'Choose File';
            chooseBtn.title = this.deeptechXlsxFile.name;
            document.getElementById('deeptech-xlsx-status').textContent = `✓ Selected: ${this._truncateFileName(this.deeptechXlsxFile.name)}`;
        }
        this.updateStartButtonState();
        requestAnimationFrame(() => this.syncUploadWindowHeights());
    }

    handleStartupXlsxFileSelect(e) {
        this.startupXlsxFile = e.target.files[0];
        if (this.startupXlsxFile) {
            const chooseBtn = document.getElementById('startup-xlsx-btn');
            chooseBtn.textContent = 'Choose File';
            chooseBtn.title = this.startupXlsxFile.name;
            document.getElementById('startup-xlsx-status').textContent = `✓ Selected: ${this._truncateFileName(this.startupXlsxFile.name)}`;
        }
        // Startup upload should not gate start button, so we only sync visual state.
        requestAnimationFrame(() => this.syncUploadWindowHeights());
    }

    _truncateFileName(filename, maxLength = 24) {
        if (!filename || filename.length <= maxLength) {
            return filename || '';
        }
        return `${filename.slice(0, maxLength - 3)}...`;
    }

    isAcceptedUploadFile(file, acceptedExtensions) {
        if (!file || !acceptedExtensions || acceptedExtensions.length === 0) {
            return true;
        }

        const lowerName = String(file.name || '').toLowerCase();
        return acceptedExtensions.some((ext) => lowerName.endsWith(ext));
    }

    syncUploadWindowHeights() {
        const updatePage = document.getElementById('update-page');
        const uploadWindows = Array.from(document.querySelectorAll('.upload-window[data-upload-type]'));

        if (!updatePage || !updatePage.classList.contains('active') || uploadWindows.length < 2) {
            uploadWindows.forEach((zone) => {
                zone.style.height = '';
            });
            return;
        }

        uploadWindows.forEach((zone) => {
            zone.style.height = 'auto';
        });

        const targetHeight = uploadWindows.reduce((maxHeight, zone) => {
            return Math.max(maxHeight, zone.getBoundingClientRect().height);
        }, 0);

        if (targetHeight > 0) {
            uploadWindows.forEach((zone) => {
                zone.style.height = `${targetHeight}px`;
            });
        }
    }

    clearUploadedFileCache() {
        this.inputCsvFile = null;
        this.deeptechXlsxFile = null;
        this.startupXlsxFile = null;
        this.isCsvUploaded = false;

        const inputCsvEl = document.getElementById('input-csv-file');
        const deeptechEl = document.getElementById('deeptech-xlsx-file');
        const startupEl = document.getElementById('startup-xlsx-file');
        if (inputCsvEl) inputCsvEl.value = '';
        if (deeptechEl) deeptechEl.value = '';
        if (startupEl) startupEl.value = '';

        const inputCsvBtn = document.getElementById('input-csv-btn');
        const deeptechBtn = document.getElementById('deeptech-xlsx-btn');
        const startupBtn = document.getElementById('startup-xlsx-btn');
        if (inputCsvBtn) {
            inputCsvBtn.textContent = 'Choose File';
            inputCsvBtn.title = '';
        }
        if (deeptechBtn) {
            deeptechBtn.textContent = 'Choose File';
            deeptechBtn.title = '';
        }
        if (startupBtn) {
            startupBtn.textContent = 'Choose File';
            startupBtn.title = '';
        }

        const inputCsvStatus = document.getElementById('input-csv-status');
        const deeptechStatus = document.getElementById('deeptech-xlsx-status');
        const startupStatus = document.getElementById('startup-xlsx-status');
        if (inputCsvStatus) inputCsvStatus.textContent = '';
        if (deeptechStatus) deeptechStatus.textContent = '';
        if (startupStatus) startupStatus.textContent = '';

        this.updateStartButtonState();
        requestAnimationFrame(() => this.syncUploadWindowHeights());
    }

    async uploadFile(file, endpoint, fileType) {
        const formData = new FormData();
        formData.append('file', file);

        // UI state locking during upload
        const submitButtonByType = {
            'input-csv': 'submit-csv-btn',
            'deeptech-xlsx': 'submit-xlsx-btn',
            'startup-xlsx': 'submit-startup-btn',
        };
        const btnId = submitButtonByType[fileType];
        const btn = document.getElementById(btnId);
        if (btn) btn.disabled = true;

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: this.authManager.getAuthHeader(),
                body: formData
            });

            if (!response.ok) {
                const data = await response.json();
                if (response.status === 401) {
                    this.authManager.logout();
                    if (window.sidebarManager) {
                        window.sidebarManager.updateAuthUI();
                    }
                    this.showMessage('Session expired. Please log in again and re-upload the file.', 'error');
                    return;
                }
                throw new Error(data.detail || data.message || 'Upload failed');
            }

            const data = await response.json();
            this.showMessage(`${file.name} uploaded successfully`, 'success');

            const statusByType = {
                'input-csv': 'input-csv-status',
                'deeptech-xlsx': 'deeptech-xlsx-status',
                'startup-xlsx': 'startup-xlsx-status',
            };
            const statusId = statusByType[fileType];
            const statusEl = document.getElementById(statusId);
            if (statusEl) {
                statusEl.textContent = `✓ Uploaded: ${this._truncateFileName(file.name)}`;
            }

            // Update state for CSV upload completion
            if (fileType === 'input-csv') {
                this.isCsvUploaded = true;
            }
            this.updateStartButtonState();
            requestAnimationFrame(() => this.syncUploadWindowHeights());
        } catch (error) {
            this.showMessage(`Upload failed: ${error.message}`, 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    updateStartButtonState() {
        const startBtn = document.getElementById('start-update-btn');
        if (startBtn && !this.isUpdating) {
            // 只要CSV上传成功，就可以点击update按钮 (Bug 8, 10)
            startBtn.disabled = !this.isCsvUploaded;
        }
    }

    async startUpdate() {
        const startBtn = document.getElementById('start-update-btn');
        if (!this.inputCsvFile) {
            this.showMessage('Please choose and upload input.csv before starting update.', 'error');
            return;
        }

        const inputCsvFilename = this.inputCsvFile.name;
        const deeptechXlsxFilename = this.deeptechXlsxFile ? this.deeptechXlsxFile.name : undefined;

        // Start update 后清空选择缓存，避免不刷新页面时沿用上次文件状态
        this.clearUploadedFileCache();

        startBtn.disabled = true; // Bug 9 lock
        startBtn.style.display = 'none'; // Bug 9 消失

        // 锁死另外两个上传按钮 (Bug 9)
        const csvBtn = document.getElementById('submit-csv-btn');
        const xlsxBtn = document.getElementById('submit-xlsx-btn');
        const startupBtn = document.getElementById('submit-startup-btn');
        if (csvBtn) csvBtn.disabled = true;
        if (xlsxBtn) xlsxBtn.disabled = true;
        if (startupBtn) startupBtn.disabled = true;

        const updateContainer = document.querySelector('.update-container');
        if (updateContainer) {
            updateContainer.classList.add('updating-layout'); // Bug 6 改变排版
            requestAnimationFrame(() => this.syncMonitorPanelHeight());
        }

        this.isUpdating = true;
        this.resetProgressDisplay();
        document.getElementById('info-status').textContent = 'Connecting...';

        try {
            const authHeader = this.authManager.getAuthHeader();
            if (!authHeader.Authorization) {
                throw new Error('Missing auth token, please log in again');
            }

            // 先建立WebSocket，避免任务已启动但前端还未订阅导致丢进度
            await this.connectWebSocket();

            // 启动更新任务
            const response = await fetch('/api/start-update', {
                method: 'POST',
                headers: {
                    ...authHeader,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    input_csv_filename: inputCsvFilename,
                    deeptech_xlsx_filename: deeptechXlsxFilename // Optional
                })
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to start update');
            }

            const data = await response.json();
            this.taskId = data.task_id;

            // 任务ID拿到后再订阅一次，确保服务端收到有效task_id
            this.subscribeToProgress();
            document.getElementById('info-status').textContent = 'Running...';
        } catch (error) {
            this.showMessage(`Failed to start update: ${error.message}`, 'error');
            startBtn.style.display = '';
            startBtn.disabled = false;
            if (csvBtn) csvBtn.disabled = false;
            if (xlsxBtn) xlsxBtn.disabled = false;
            if (startupBtn) startupBtn.disabled = false;
            if (updateContainer) {
                updateContainer.classList.remove('updating-layout');
            }
            const monitorSection = document.querySelector('.monitor-section');
            if (monitorSection) {
                monitorSection.style.height = '';
            }
            this.isUpdating = false;
            this.updateStartButtonState();
        }
    }

    syncMonitorPanelHeight() {
        const updateContainer = document.querySelector('.update-container');
        const monitorSection = document.querySelector('.monitor-section');
        if (!updateContainer || !monitorSection || !updateContainer.classList.contains('updating-layout')) {
            if (monitorSection) {
                monitorSection.style.height = '';
            }
            return;
        }

        const startBtn = document.getElementById('start-update-btn');
        const submitXlsxBtn = document.getElementById('submit-xlsx-btn');
        const submitStartupBtn = document.getElementById('submit-startup-btn');

        // Running: align with Submit XLSX; Completed: align with visible Start button.
        let anchor = null;
        if (startBtn && startBtn.offsetParent !== null) {
            anchor = startBtn;
        } else {
            const runningAnchors = [submitXlsxBtn, submitStartupBtn].filter((node) => node && node.offsetParent !== null);
            if (runningAnchors.length > 0) {
                anchor = runningAnchors.reduce((lowest, node) => {
                    if (!lowest) return node;
                    return node.getBoundingClientRect().bottom > lowest.getBoundingClientRect().bottom ? node : lowest;
                }, null);
            }
        }
        if (!anchor) return;

        const monitorTop = monitorSection.getBoundingClientRect().top;
        const anchorBottom = anchor.getBoundingClientRect().bottom;
        const desiredHeight = Math.max(220, Math.round(anchorBottom - monitorTop));
        monitorSection.style.height = `${desiredHeight}px`;
    }

    connectWebSocket() {
        if (this.wsSocket && this.wsSocket.readyState === WebSocket.OPEN) {
            return Promise.resolve();
        }

        if (this.wsSocket && this.wsSocket.readyState === WebSocket.CONNECTING) {
            return new Promise((resolve, reject) => {
                const handleOpen = () => {
                    this.subscribeToProgress();
                    resolve();
                };
                const handleError = () => reject(new Error('WebSocket connection error'));
                this.wsSocket.addEventListener('open', handleOpen, { once: true });
                this.wsSocket.addEventListener('error', handleError, { once: true });
            });
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const token = this.authManager.syncTokenFromStorage();
        if (!token) {
            return Promise.reject(new Error('Missing auth token'));
        }
        const wsUrl = `${protocol}//${window.location.host}/ws/${token}`;

        return new Promise((resolve, reject) => {
            this.wsSocket = new WebSocket(wsUrl);

            this.wsSocket.onopen = () => {
                console.log('WebSocket connected');
                this.subscribeToProgress();
                resolve();
            };

            this.wsSocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleWSMessage(message);
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            this.wsSocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.showMessage('WebSocket connection error', 'error');
                reject(new Error('WebSocket connection error'));
            };

            this.wsSocket.onclose = () => {
                console.log('WebSocket disconnected');
                // 自动重连（指数退避，最多3次）
                if (this.isUpdating) {
                    this.attemptWebSocketReconnect(1);
                }
            };
        });
    }

    subscribeToProgress() {
        if (!this.wsSocket || this.wsSocket.readyState !== WebSocket.OPEN) {
            return;
        }
        this.wsSocket.send(JSON.stringify({
            type: 'subscribe_progress',
            task_id: this.taskId
        }));
    }

    async attemptWebSocketReconnect(attempt) {
        if (attempt > 3) {
            this.showMessage('WebSocket connection lost', 'error');
            return;
        }

        const delayMs = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
        console.log(`Attempting WebSocket reconnect in ${delayMs}ms (attempt ${attempt})`);

        await new Promise(resolve => setTimeout(resolve, delayMs));
        try {
            await this.connectWebSocket();
        } catch (error) {
            await this.attemptWebSocketReconnect(attempt + 1);
        }
    }

    handleWSMessage(message) {
        const { type, data, timestamp } = message;

        switch (type) {
            case 'progress':
                this.updateProgress(data);
                break;
            case 'error':
                this.handleError(data);
                break;
            case 'completed':
                this.handleCompletion(data);
                break;
            case 'log':
                this.handleLog(data);
                break;
        }
    }

    updateProgress(data) {
        const { progress_pct, current_stage, current_professor } = data;

        // 更新进度条
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        progressFill.classList.remove('completed');
        progressFill.style.width = `${progress_pct}%`;
        progressText.textContent = `${Math.round(progress_pct)}%`;

        // 更新信息框
        document.getElementById('info-stage').textContent = this._formatStageName(current_stage) || 'Processing';
        document.getElementById('info-professor').textContent = current_professor || '-';
        document.getElementById('info-status').textContent = 'Running...';
    }

    handleError(data) {
        const { error_msg, current_stage, current_professor } = data;
        console.error('Update error:', error_msg);
        this.showMessage(`Error: ${error_msg}`, 'error');

        const startBtn = document.getElementById('start-update-btn');
        const csvBtn = document.getElementById('submit-csv-btn');
        const xlsxBtn = document.getElementById('submit-xlsx-btn');
        const startupBtn = document.getElementById('submit-startup-btn');

        this.isUpdating = false;
        if (this.wsSocket) {
            this.wsSocket.close();
            this.wsSocket = null;
        }
        if (startBtn) {
            startBtn.style.display = '';
            startBtn.disabled = false;
        }
        if (csvBtn) csvBtn.disabled = false;
        if (xlsxBtn) xlsxBtn.disabled = false;
        if (startupBtn) startupBtn.disabled = false;

        this.syncMonitorPanelHeight();
        this.updateStartButtonState();

        document.getElementById('info-stage').textContent = this._formatStageName(current_stage) || 'Failed';
        document.getElementById('info-professor').textContent = current_professor || '-';
        document.getElementById('info-status').textContent = 'Failed';

        // 详细记录到日志
        console.error(`Failed at stage "${current_stage}" for professor "${current_professor}": ${error_msg}`);
    }

    handleCompletion(data) {
        const { markdown_content, summary_stats } = data;

        this.isUpdating = false;

        // 关闭WebSocket
        if (this.wsSocket) {
            this.wsSocket.close();
            this.wsSocket = null;
        }

        const startBtn = document.getElementById('start-update-btn');
        const csvBtn = document.getElementById('submit-csv-btn');
        const xlsxBtn = document.getElementById('submit-xlsx-btn');
        const startupBtn = document.getElementById('submit-startup-btn');

        if (startBtn) {
            startBtn.style.display = '';
            startBtn.disabled = false;
        }
        if (csvBtn) csvBtn.disabled = false;
        if (xlsxBtn) xlsxBtn.disabled = false;
        if (startupBtn) startupBtn.disabled = false;

        // 显示报告
        this.showReport(markdown_content, summary_stats);
        this.syncMonitorPanelHeight();
        this.updateStartButtonState();
        this.showMessage('Update completed successfully!', 'success');
    }

    showReport(markdownContent, summaryStats) {
        const reportContainer = document.getElementById('report-container');
        const reportContent = document.getElementById('report-content');
        const progressFill = document.getElementById('progress-fill');

        reportContainer.classList.remove('hidden');
        reportContent.textContent = markdownContent;

        // 更新进度显示
        progressFill.style.width = '100%';
        progressFill.classList.add('completed');
        document.getElementById('progress-text').textContent = '100%';
        document.getElementById('info-stage').textContent = 'Completed';
        document.getElementById('info-status').textContent = 'SuccessFul';
        requestAnimationFrame(() => this.syncMonitorPanelHeight());
    }

    downloadReport() {
        const content = document.getElementById('report-content').textContent;
        if (!content) return;

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `professor_update_report_${timestamp}.md`;

        const blob = new Blob([content], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    resetProgressDisplay() {
        const progressFill = document.getElementById('progress-fill');
        progressFill.classList.remove('completed');
        progressFill.style.width = '0%';
        document.getElementById('progress-text').textContent = '0%';
        document.getElementById('info-stage').textContent = 'Initializing...';
        document.getElementById('info-professor').textContent = '-';
        document.getElementById('info-status').textContent = 'Starting...';
        document.getElementById('report-container').classList.add('hidden');
    }

    handleLog(data) {
        console.log('[Update Log]', data.log_msg);
        const line = String(data.log_msg || '').trim();
        if (!line) return;
        document.getElementById('info-status').textContent = line.slice(0, 56);
    }

    showMessage(text, type = 'info') {
        // 使用toast通知或简单alert
        if (type === 'error') console.error(text);
        else if (type === 'success') console.log('[✓]', text);
        else console.info(text);
    }

    _formatStageName(stage) {
        const stageNames = {
            'scrape_info': 'Scraping Professor Information',
            'scrape_publication': 'Scraping Publications',
            'scrape_project': 'Scraping Projects',
            'merging_reports': 'Merging Reports'
        };
        return stageNames[stage] || stage;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const startupSearchManager = new StartupSearchManager();
    window.startupSearchManager = startupSearchManager;

    const authManager = new AuthManager();
    window.authManager = authManager;

    const sidebarManager = new SidebarManager(authManager);
    window.sidebarManager = sidebarManager;

    const updateManager = new DatabaseUpdateManager(authManager);
    window.updateManager = updateManager;

    // Bug 11: 级联逻辑 alpha + beta = 1
    const professorAlphaInput = document.getElementById('professorAlpha');
    const professorBetaInput = document.getElementById('professorBeta');

    const bindAlphaBetaPair = (alphaEl, betaEl) => {
        if (!alphaEl || !betaEl) return;

        alphaEl.addEventListener('input', (e) => {
            let val = parseFloat(e.target.value);
            if (isNaN(val)) return;
            if (val > 1) val = 1;
            if (val < 0) val = 0;
            betaEl.value = (1 - val).toFixed(2);
        });

        betaEl.addEventListener('input', (e) => {
            let val = parseFloat(e.target.value);
            if (isNaN(val)) return;
            if (val > 1) val = 1;
            if (val < 0) val = 0;
            alphaEl.value = (1 - val).toFixed(2);
        });
    };

    bindAlphaBetaPair(professorAlphaInput, professorBetaInput);
});

// Premium UI Systems - Added via Skill
document.addEventListener("DOMContentLoaded", () => {
    initPremiumUI();
});

function initPremiumUI() {
    if (typeof gsap !== "undefined" && typeof ScrollTrigger !== "undefined") {
        gsap.registerPlugin(ScrollTrigger);
    }

    // 1. Lenis Smooth Scroll
    if (typeof Lenis !== "undefined") {
        const lenis = new Lenis({
            duration: 0.8, // reduced duration for snappier feel
            easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
            orientation: "vertical",
            gestureOrientation: "vertical",
            smoothWheel: true,
            wheelMultiplier: 1,
            smoothTouch: false,
            touchMultiplier: 2,
            infinite: false,
        });

        const addLenisPrevent = () => {
            document.querySelectorAll(".modal-content, .deeptech-modal-content, .startup-modal-content, .deeptech-chip-list.has-scroll").forEach(el => {
                el.setAttribute("data-lenis-prevent", "");
            });
        };
        addLenisPrevent();
        const observer = new MutationObserver((mutations) => {
            for (let m of mutations) {
                if (m.addedNodes.length) addLenisPrevent();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });

        function raf(time) {
            lenis.raf(time);
            requestAnimationFrame(raf);
        }
        requestAnimationFrame(raf);
    }

    // 2. Preloader & Intro Animation (single seamless timeline)
    const preloader = document.getElementById("preloader");
    const activePage = document.querySelector(".page.active");

    if (typeof gsap !== "undefined") {
        primeHeroIntro(activePage);
    }

    if (preloader && typeof gsap !== "undefined") {
        window.__preloaderHandled = true;

        const fallbackTimer = window.setTimeout(() => {
            if (preloader.parentElement) {
                preloader.remove();
            }
            animateHero({ page: document.querySelector(".page.active") });
        }, 5200);

        gsap.timeline({
            defaults: { overwrite: "auto" },
            onComplete: () => {
                window.clearTimeout(fallbackTimer);
                if (preloader.parentElement) {
                    preloader.remove();
                }
            },
        })
            .to(".preloader-brand", {
                opacity: 1,
                y: 0,
                duration: 0.55,
                ease: "power2.out",
            })
            .to(".preloader-progress", {
                opacity: 1,
                duration: 0.2,
                ease: "power1.out",
            }, "<")
            .to(".preloader-bar", {
                width: "100%",
                duration: 0.85,
                ease: "power3.inOut",
            }, "-=0.04")
            .to(preloader, {
                yPercent: -100,
                duration: 0.62,
                ease: "expo.inOut",
            }, "-=0.06")
            .add(() => {
                animateHero({ page: document.querySelector(".page.active") });
            }, "-=0.18");
    } else {
        window.__preloaderHandled = true;
        animateHero({ page: activePage });
    }
}

function primeHeroIntro(activePage) {
    if (typeof gsap === "undefined" || !activePage) return;

    const isSearchHeroPage =
        activePage.id === "professor-page" || activePage.id === "startup-page";
    const titleOffset = isSearchHeroPage ? "132%" : "105%";
    const fallbackTitleY = isSearchHeroPage ? 30 : 22;
    const subtitleY = isSearchHeroPage ? 30 : 16;
    const heroCardY = isSearchHeroPage ? 44 : 24;

    const h1 = activePage.querySelector("h1");
    const subtitle = activePage.querySelector(".subtitle");
    const heroCard = activePage.querySelector(".hero-card");

    if (h1) {
        if (typeof SplitType !== "undefined") {
            const existingChars = h1.querySelectorAll(".char");
            if (existingChars.length === 0) {
                h1._splitInstance = new SplitType(h1, { types: "chars, words" });
            }
            const chars = h1.querySelectorAll(".char");
            if (chars.length > 0) {
                gsap.set(chars, { y: titleOffset, opacity: 0, rotateZ: 4 });
            } else {
                gsap.set(h1, { y: fallbackTitleY, opacity: 0 });
            }
        } else {
            gsap.set(h1, { y: fallbackTitleY, opacity: 0 });
        }
    }

    if (subtitle) {
        gsap.set(subtitle, { y: subtitleY, opacity: 0 });
    }

    if (heroCard) {
        gsap.set(heroCard, { y: heroCardY, opacity: 0 });
    }
}

function animateHero(options = {}) {
    if (typeof gsap === "undefined") return null;

    const activePage = options.page || document.querySelector(".page.active");
    if (!activePage) return null;

    const isSearchHeroPage =
        activePage.id === "professor-page" || activePage.id === "startup-page";
    const titleDuration = isSearchHeroPage ? 0.92 : 0.82;
    const subtitleStart = isSearchHeroPage ? 0.1 : 0.06;
    const subtitleDuration = isSearchHeroPage ? 0.68 : 0.62;
    const cardStart = isSearchHeroPage ? 0.16 : 0.1;
    const cardDuration = isSearchHeroPage ? 0.76 : 0.68;

    const h1 = activePage.querySelector("h1");
    const subtitle = activePage.querySelector(".subtitle");
    const heroCard = activePage.querySelector(".hero-card");

    primeHeroIntro(activePage);

    const tl = gsap.timeline({ defaults: { overwrite: "auto" } });

    if (h1) {
        const chars = h1.querySelectorAll(".char");
        if (chars.length > 0) {
            tl.to(chars, {
                y: "0%",
                opacity: 1,
                rotateZ: 0,
                duration: titleDuration,
                ease: "expo.out",
                stagger: 0.012,
            }, 0);
        } else {
            tl.to(h1, {
                y: 0,
                opacity: 1,
                duration: titleDuration,
                ease: "expo.out",
            }, 0);
        }
    }

    if (subtitle) {
        tl.to(subtitle, {
            y: 0,
            opacity: 1,
            duration: subtitleDuration,
            ease: "power2.out",
        }, subtitleStart);
    }

    if (heroCard) {
        tl.to(heroCard, {
            y: 0,
            opacity: 1,
            duration: cardDuration,
            ease: "expo.out",
        }, cardStart);
    }

    return tl;
}

// Hook into results rendering to apply staggered scroll animations
if (typeof renderProfessorResults !== "undefined") {
    const originalRenderProfessorResults = renderProfessorResults;
    renderProfessorResults = function (results, query, extractedKeywords) {
        originalRenderProfessorResults(results, query, extractedKeywords);
        animateResultsGrid(document.getElementById("professorResultsList"));
    };
}

if (typeof StartupSearchManager !== "undefined" && StartupSearchManager.prototype.renderStartupResults) {
    const originalRenderStartupResults = StartupSearchManager.prototype.renderStartupResults;
    StartupSearchManager.prototype.renderStartupResults = function (items, extractedKeywords) {
        originalRenderStartupResults.call(this, items, extractedKeywords);
        animateResultsGrid(this.resultsList);
    };
}

function animateResultsGrid(gridContainer) {
    if (!gridContainer || typeof gsap === "undefined" || typeof ScrollTrigger === "undefined") return;
    const cards = gridContainer.querySelectorAll(".result-card");

    gsap.set(cards, { y: 30, opacity: 0, scale: 0.98 });

    ScrollTrigger.batch(cards, {
        interval: 0.1,
        batchMax: 5,
        onEnter: batch => gsap.to(batch, {
            y: 0,
            opacity: 1,
            scale: 1,
            duration: 0.6,
            ease: "back.out(1.1)",
            stagger: 0.05,
            overwrite: true
        })
    });
}

// Auto trigger hero animations on page switch
document.addEventListener("DOMContentLoaded", () => {
    // We hook the navigation logic so that each page switch plays the polished entrance
    if (typeof SidebarManager !== "undefined" && SidebarManager.prototype.navigateToPage) {
        const originalNavigateToPage = SidebarManager.prototype.navigateToPage;
        SidebarManager.prototype.navigateToPage = function (pageType) {
            originalNavigateToPage.call(this, pageType);
            const activePage = document.querySelector(".page.active");
            if (typeof animateHero === "function") animateHero({ page: activePage });
        };
    }
});
