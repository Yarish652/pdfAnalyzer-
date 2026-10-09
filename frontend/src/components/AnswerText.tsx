import { Fragment, type ReactNode } from "react";

type Props = {
  text: string;
  sourceIds: Set<number>;
  activeSource: number | null;
  onCite: (id: number) => void;
};

type Block =
  | { kind: "paragraph"; text: string }
  | { kind: "list"; ordered: boolean; items: string[] };

const BULLET = /^\s*[-*•]\s+/;
const NUMBERED = /^\s*\d+[.)]\s+/;
// Bold, inline code, or a run of citations such as [1], [1][3] or [1, 2].
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|(?:\[\d+(?:\s*,\s*\d+)*\])+)/g;

/**
 * Renders model output safely as React elements (never as HTML): paragraphs,
 * lists, bold, inline code, and citation chips linked to the sources.
 */
export function AnswerText({ text, sourceIds, activeSource, onCite }: Props) {
  return (
    <>
      {parseBlocks(text).map((block, index) =>
        block.kind === "paragraph" ? (
          <p key={index}>{renderInline(block.text, sourceIds, activeSource, onCite)}</p>
        ) : block.ordered ? (
          <ol key={index}>
            {block.items.map((item, itemIndex) => (
              <li key={itemIndex}>{renderInline(item, sourceIds, activeSource, onCite)}</li>
            ))}
          </ol>
        ) : (
          <ul key={index}>
            {block.items.map((item, itemIndex) => (
              <li key={itemIndex}>{renderInline(item, sourceIds, activeSource, onCite)}</li>
            ))}
          </ul>
        ),
      )}
    </>
  );
}

function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  let paragraph: string[] = [];

  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ kind: "paragraph", text: paragraph.join(" ") });
    paragraph = [];
  };

  for (const line of text.split("\n")) {
    const ordered = NUMBERED.test(line);
    if (BULLET.test(line) || ordered) {
      flushParagraph();
      const item = line.replace(ordered ? NUMBERED : BULLET, "");
      const last = blocks[blocks.length - 1];
      if (last?.kind === "list" && last.ordered === ordered) last.items.push(item);
      else blocks.push({ kind: "list", ordered, items: [item] });
    } else if (!line.trim()) {
      flushParagraph();
    } else {
      paragraph.push(line.trim());
    }
  }
  flushParagraph();
  return blocks;
}

function renderInline(
  text: string,
  sourceIds: Set<number>,
  activeSource: number | null,
  onCite: (id: number) => void,
): ReactNode[] {
  return text.split(INLINE).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }
    if (/^\[\d/.test(part)) {
      const ids = [...part.matchAll(/\d+/g)].map((match) => Number(match[0]));
      // Leave citations to passages that do not exist as plain text.
      if (!ids.every((id) => sourceIds.has(id))) return <Fragment key={index}>{part}</Fragment>;
      return (
        <span key={index} className="citations">
          {ids.map((id) => (
            <button
              key={id}
              type="button"
              className={`citation ${activeSource === id ? "is-active" : ""}`}
              onClick={() => onCite(id)}
              aria-label={`Show source ${id}`}
            >
              {id}
            </button>
          ))}
        </span>
      );
    }
    return <Fragment key={index}>{part}</Fragment>;
  });
}
