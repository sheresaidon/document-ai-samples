from __future__ import annotations

from collections.abc import Sequence

from google.api_core.client_options import ClientOptions
from google.cloud import documentai  # type: ignore

project_id = 'bright-black-ai'
location = 'us' # Format is 'us' or 'eu'
processor_id = '18c798abe7fde749' # Create processor before running sample
file_path = '/Users/sheresaidon/Downloads/nish/Bloodwork 2.pdf'
mime_type = 'application/pdf' # Refer to https://cloud.google.com/document-ai/docs/file-types for supported file types

def process_document_form_sample(
    project_id: str, location: str, processor_id: str, file_path: str, mime_type: str
) -> None:
    # Online processing request to Document AI
    document = process_document(
        project_id, location, processor_id, file_path, mime_type
    )

    text = document.text
    print(f"Full document text: {repr(text)}\n")
    print(f"There are {len(document.pages)} page(s) in this document.")

    for page in document.pages:
        print(f"\n\n**** Page {page.page_number} ****")

        print(f"\nFound {len(page.tables)} table(s):")
        for table in page.tables:
            num_collumns = len(table.header_rows[0].cells)
            num_rows = len(table.body_rows)
            print(f"Table with {num_collumns} columns and {num_rows} rows:")

            print("Columns:")
            print_table_rows(table.header_rows, text)
            print("Table body data:")
            print_table_rows(table.body_rows, text)

        print(f"\nFound {len(page.form_fields)} form field(s):")
        for field in page.form_fields:
            name = layout_to_text(field.field_name, text)
            value = layout_to_text(field.field_value, text)
            print(f"    * {repr(name.strip())}: {repr(value.strip())}")

        print("\nPrinting lines for the text that is not in a table:")
        print_lines(page.lines, text, page.tables)

def process_document(
    project_id: str, location: str, processor_id: str, file_path: str, mime_type: str
) -> documentai.Document:
    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")

    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    name = client.processor_path(project_id, location, processor_id)

    with open(file_path, "rb") as image:
        image_content = image.read()

    raw_document = documentai.RawDocument(content=image_content, mime_type=mime_type)

    request = documentai.ProcessRequest(name=name, raw_document=raw_document)

    result = client.process_document(request=request)

    return result.document

def print_table_rows(
    table_rows: Sequence[documentai.Document.Page.Table.TableRow], text: str
) -> None:
    for table_row in table_rows:
        row_text = ""
        for cell in table_row.cells:
            cell_text = layout_to_text(cell.layout, text)
            row_text += f"{repr(cell_text.strip())} | "
        print(row_text)

def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    response = ""
    for segment in layout.text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response

def print_lines(lines: Sequence[documentai.Document.Page.Line], text: str, tables: Sequence[documentai.Document.Page.Table]) -> str:
    print(f"    {len(lines)} lines detected:")

    table_bboxes = []
    for table in tables:
        all_rows = list(table.header_rows)
        all_rows.extend(table.body_rows)
        for row in all_rows:
            for cell in row.cells:
                table_bboxes.append(cell.layout.bounding_poly)

    def is_line_in_table(line_bbox, table_bboxes):
        for table_bbox in table_bboxes:
            if (
                line_bbox.vertices[0].x >= table_bbox.vertices[0].x
                and line_bbox.vertices[0].y >= table_bbox.vertices[0].y
                and line_bbox.vertices[2].x <= table_bbox.vertices[2].x
                and line_bbox.vertices[2].y <= table_bbox.vertices[2].y
            ):
                return True
        return False

    lines = [line for line in lines if not is_line_in_table(line.layout.bounding_poly, table_bboxes)]

    sorted_lines = sorted(lines, key=lambda line: line.layout.bounding_poly.vertices[0].y)

    grouped_lines = []
    for i, line in enumerate(sorted_lines):
        line_text = layout_to_text(line.layout, text)
        line_x = line.layout.bounding_poly.vertices[0].x
        line_y = line.layout.bounding_poly.vertices[0].y

        if i % 3 == 0:
            y_diffs = []
            for j in range(i + 1, min(i + 11, len(sorted_lines))):
                prev_line_y = sorted_lines[j - 1].layout.bounding_poly.vertices[0].y
                curr_line_y = sorted_lines[j].layout.bounding_poly.vertices[0].y
                y_diffs.append(curr_line_y - prev_line_y)

            if y_diffs:
                avg_y_diff = sum(y_diffs) / len(y_diffs)

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

    for group in grouped_lines:
        group.sort(key=lambda x: x[1])

    lines_text = ""
    for group in grouped_lines:
        line_texts = []
        for i, line in enumerate(group):
            line_text = line[0].replace('\n', ' ')
            if i > 0:
                prev_line_x = group[i - 1][1]
                curr_line_x = line[1]
                space_count = max(int((curr_line_x - prev_line_x) / 100), 4)
                line_texts.append(" " * space_count + line_text)
            else:
                line_texts.append(line_text)
        lines_text += "".join(line_texts) + "\n"
    print(lines_text)
    return lines_text

process_document_form_sample(
      project_id=project_id,    location=location,
    processor_id=processor_id,     file_path=file_path,     mime_type=mime_type
)