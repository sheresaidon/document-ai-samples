from typing import Sequence

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1beta3 as documentai


def process_document_ocr_sample(
    project_id: str,
    location: str,
    processor_id: str,
    processor_version: str,
    file_path: str,
    mime_type: str,
    enable_native_pdf_parsing: bool,
) -> None:
    # Online processing request to Document AI
    document = process_document(
        project_id,
        location,
        processor_id,
        processor_version,
        file_path,
        mime_type,
        enable_native_pdf_parsing,
    )

    # For a full list of Document object attributes, please reference this page:
    # https://cloud.google.com/python/docs/reference/documentai/latest/google.cloud.documentai_v1.types.Document

    text = document.text
    print(f"Full document text: {text}\n")
    print(f"There are {len(document.pages)} page(s) in this document.\n")

    for page in document.pages:
        print(f"Page {page.page_number}:")
        print_page_dimensions(page.dimension)
        print_detected_langauges(page.detected_languages)
        print_paragraphs(page.paragraphs, text)
        print_blocks(page.blocks, text)
        print_lines(page.lines, text)
        print_tokens(page.tokens, text)

        # Currently supported in version pretrained-ocr-v1.1-2022-09-12
        if page.image_quality_scores:
            print_image_quality_scores(page.image_quality_scores)


def process_document(
    project_id: str,
    location: str,
    processor_id: str,
    processor_version: str,
    file_path: str,
    mime_type: str,
    enable_native_pdf_parsing: bool,
) -> documentai.Document:
    # You must set the api_endpoint if you use a location other than 'us'.
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    # The full resource name of the processor version
    # e.g. projects/{project_id}/locations/{location}/processors/{processor_id}/processorVersions/{processor_version_id}
    # You must create processors before running sample code.
    name = client.processor_version_path(
        project_id, location, processor_id, processor_version
    )

    # Read the file into memory
    with open(file_path, "rb") as image:
        image_content = image.read()

    # Load Binary Data into Document AI RawDocument Object
    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)

    process_options = documentai.ProcessOptions(
        ocr_config=documentai.OcrConfig(
            enable_native_pdf_parsing=enable_native_pdf_parsing
        )
    )

    # Configure the process request
    request = documentai.ProcessRequest(
        name=name, raw_document=raw_document, process_options=process_options
    )

    result = client.process_document(request=request)

    return result.document


def print_page_dimensions(dimension: documentai.Document.Page.Dimension) -> None:
    print(f"    Width: {str(dimension.width)}")
    print(f"    Height: {str(dimension.height)}")


def print_detected_langauges(
    detected_languages: Sequence[documentai.Document.Page.DetectedLanguage],
) -> None:
    print("    Detected languages:")
    for lang in detected_languages:
        code = lang.language_code
        print(f"        {code} ({lang.confidence:.1%} confidence)")


def print_paragraphs(
    paragraphs: Sequence[documentai.Document.Page.Paragraph], text: str
) -> None:
    print(f"    {len(paragraphs)} paragraphs detected:")
    first_paragraph_text = layout_to_text(paragraphs[0].layout, text)
    print(f"        First paragraph text: {repr(first_paragraph_text)}")
    last_paragraph_text = layout_to_text(paragraphs[-1].layout, text)
    print(f"        Last paragraph text: {repr(last_paragraph_text)}")


def print_blocks(blocks: Sequence[documentai.Document.Page.Block], text: str) -> None:
    print(f"    {len(blocks)} blocks detected:")
    first_block_text = layout_to_text(blocks[0].layout, text)
    print(f"        First text block: {repr(first_block_text)}")
    last_block_text = layout_to_text(blocks[-1].layout, text)
    print(f"        Last text block: {repr(last_block_text)}")


def print_lines(lines: Sequence[documentai.Document.Page.Line], text: str) -> None:
    print(f"    {len(lines)} lines detected:")

    # Sort lines by their y position
    sorted_lines = sorted(lines, key=lambda line: line.layout.bounding_poly.vertices[0].y)

    # Group lines with similar y positions
    grouped_lines = []
    for i, line in enumerate(sorted_lines):
        line_text = layout_to_text(line.layout, text)
        line_x = line.layout.bounding_poly.vertices[0].x
        line_y = line.layout.bounding_poly.vertices[0].y

        # Calculate the average difference in y positions between consecutive lines for every 10 lines
        if i % 3 == 0:
            y_diffs = []
            for j in range(i + 1, min(i + 11, len(sorted_lines))):
                prev_line_y = sorted_lines[j - 1].layout.bounding_poly.vertices[0].y
                curr_line_y = sorted_lines[j].layout.bounding_poly.vertices[0].y
                y_diffs.append(curr_line_y - prev_line_y)
            avg_y_diff = sum(y_diffs) / len(y_diffs)

            # Set the tolerance value based on the average y difference
            tolerance = avg_y_diff / 2
            print(f"        Average y difference: {avg_y_diff:.1f}")
            
        added_to_group = False
        for group in grouped_lines:
            group_y = group[0][2]
            if abs(line_y - group_y) <= tolerance:
                group.append((line_text, line_x, line_y))
                added_to_group = True
                break

        if not added_to_group:
            grouped_lines.append([(line_text, line_x, line_y)])

    # Sort lines in each group by their x position
    for group in grouped_lines:
        group.sort(key=lambda x: x[1])

    # Print grouped lines with adjusted spacing
    for group in grouped_lines:
        line_texts = []
        for i, line in enumerate(group):
            line_text = line[0].replace('\n', ' ')  # Replace newline characters with spaces
            if i > 0:
                prev_line_x = group[i - 1][1]
                curr_line_x = line[1]
                space_count = max(int((curr_line_x - prev_line_x) / 100), 4)  # Adjust this value to control the spacing
                line_texts.append(" " * space_count + line_text)
            else:
                line_texts.append(line_text)
        print("".join(line_texts))

def print_tokens(tokens: Sequence[documentai.Document.Page.Token], text: str) -> None:
    print(f"    {len(tokens)} tokens detected:")
    first_token_text = layout_to_text(tokens[0].layout, text)
    first_token_break_type = tokens[0].detected_break.type_.name
    print(f"        First token text: {repr(first_token_text)}")
    print(f"        First token break type: {repr(first_token_break_type)}")
    last_token_text = layout_to_text(tokens[-1].layout, text)
    last_token_break_type = tokens[-1].detected_break.type_.name
    print(f"        Last token text: {repr(last_token_text)}")
    print(f"        Last token break type: {repr(last_token_break_type)}")


def print_image_quality_scores(
    image_quality_scores: documentai.Document.Page.ImageQualityScores,
) -> None:
    print(f"    Quality score: {image_quality_scores.quality_score:.1%}")
    print("    Detected defects:")

    for detected_defect in image_quality_scores.detected_defects:
        print(f"        {detected_defect.type_}: {detected_defect.confidence:.1%}")


def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    Document AI identifies text in different parts of the document by their
    offsets in the entirety of the document's text. This function converts
    offsets to a string.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in layout.text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index].replace('\n', ' ')
    return response


# TODO(developer): Edit these variables before running the sample.
project_id = "bright-black-ai"
location = "us"  # Format is 'us' or 'eu'
processor_id = "9db77bc6dadc8719"  # Create processor before running sample
processor_version = "pretrained-ocr-v1.2-2022-11-10"
file_path = "/Users/sheresaidon/Downloads/Bloodwork_PII.pdf"
# file_path = "/Users/sheresaidon/Downloads/test2.pdf"

mime_type = "application/pdf"  # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types
enable_native_pdf_parsing = True

process_document_ocr_sample(
    project_id=project_id,
    location=location,
    processor_id=processor_id,
    processor_version=processor_version,
    file_path=file_path,
    mime_type=mime_type,
    enable_native_pdf_parsing=enable_native_pdf_parsing,
)
