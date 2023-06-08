
from typing import List, Sequence
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
import os
from PyPDF2 import PdfWriter, PdfReader

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
        print_lines(page.lines, text, page.tables)
        print_tokens(page.tokens, text)

        # Currently supported in version pretrained-ocr-v1.1-2022-09-12
        if page.image_quality_scores:
            print_image_quality_scores(page.image_quality_scores)
    
    save_annotated_pdf(document, input_pdf_path, output_pdf_path)


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


def print_lines(lines: Sequence[documentai.Document.Page.Line], text: str, tables: Sequence[documentai.Document.Page.Table]) -> None:
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

    # Print tables
    if tables:
        print("                 Tables:")
        for table_idx, table in enumerate(tables):
            print(f"        Table {table_idx + 1}:")
            header_row_values = get_table_data(table.header_rows, text)
            body_row_values = get_table_data(table.body_rows, text)
            all_rows = header_row_values + body_row_values
            for row in all_rows:
                print("\t".join(row))


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
        response += text[start_index:end_index].replace('\n', '  ')
    return response

def get_table_data(
    rows: Sequence[documentai.Document.Page.Table.TableRow], text: str
) -> List[List[str]]:
    """
    Get Text data from table rows
    """
    all_values: List[List[str]] = []
    for row in rows:
        current_row_values: List[str] = []
        for cell in row.cells:
            current_row_values.append(
                text_anchor_to_text(cell.layout.text_anchor, text)
            )
        all_values.append(current_row_values)
    return all_values

def text_anchor_to_text(text_anchor: documentai.Document.TextAnchor, text: str) -> str:
    """
    Document AI identifies table data by their offsets in the entirity of the
    document's text. This function converts offsets to a string.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response.strip().replace("\n", "  ")

def save_annotated_pdf(document: documentai.Document, input_pdf_path: str, output_pdf_path: str) -> None:
    # Iterate through the pages and layout elements
    for page in document.pages:
        # Calculate the size of the output image based on the dimensions of the PDF page
        page_width = int(page.dimension.width)
        page_height = int(page.dimension.height)
        size = (page_width, page_height)

        # Convert the input PDF to an image with the specified size
        pdf_image = convert_from_path(input_pdf_path, size=size, first_page=page.page_number, last_page=page.page_number)[0]
        draw = ImageDraw.Draw(pdf_image)
        # Helper function to draw bounding boxes
        def draw_bounding_box(layout, color):
            vertices = layout.bounding_poly.vertices
            if len(vertices) == 4:
                draw.polygon([
                    (vertices[0].x, vertices[0].y),
                    (vertices[1].x, vertices[1].y),
                    (vertices[2].x, vertices[2].y),
                    (vertices[3].x, vertices[3].y)
                ], outline=color,
                             width=3)

        # Draw bounding boxes for paragraphs, blocks, lines, and tokens
        for paragraph in page.paragraphs:
            draw_bounding_box(paragraph.layout, 'red')
        for block in page.blocks:
            draw_bounding_box(block.layout, 'blue')
        for line in page.lines:
            draw_bounding_box(line.layout, 'green')
        for token in page.tokens:
            draw_bounding_box(token.layout, 'yellow')


        # Save the annotated image as a PDF
        pdf_image.save(f"{output_pdf_path[:-4]}_page{page.page_number}.pdf", "PDF")

    # Combine the individual PDF pages into a single PDF
    pdf_writer = PdfWriter()
    for page_number in range(1, len(document.pages) + 1):
        with open(f"{output_pdf_path[:-4]}_page{page_number}.pdf", "rb") as page_file:
            pdf_writer.add_page(PdfReader(page_file).pages[0])

    # Save the combined PDF
    with open(output_pdf_path, "wb") as output_file:
        pdf_writer.write(output_file)

    # Remove the individual PDF pages
    for page_number in range(1, len(document.pages) + 1):
        os.remove(f"{output_pdf_path[:-4]}_page{page_number}.pdf")

# TODO(developer): Edit these variables before running the sample.
project_id = "bright-black-ai"
location = "us"  # Format is 'us' or 'eu'
processor_id = "9db77bc6dadc8719"  # Create processor before running sample
processor_version = "pretrained-ocr-v1.2-2022-11-10"
# file_path = "/Users/sheresaidon/Downloads/Bloodwork_PII.pdf"
file_path = "/Users/sheresaidon/Downloads/nish/Provider Notes from ER visit.pdf"

# file_path = "/Users/sheresaidon/Downloads/2069394423-1098-1_1_2023.pdf"
input_pdf_path = file_path
file_name, file_extension = os.path.splitext(file_path)
output_pdf_path = f"{file_name}_annotated{file_extension}"
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
