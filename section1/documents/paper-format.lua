-- Allow long repository paths to break in LaTeX while retaining Markdown code spans.
function Code(el)
  if FORMAT:match("latex") and el.text:find("/", 1, true) then
    return pandoc.RawInline("latex", "\\path{" .. el.text .. "}")
  end
end
