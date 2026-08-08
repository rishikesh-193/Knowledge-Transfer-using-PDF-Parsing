import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pypdf
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.ingestion import (
    load_and_split_pdf,
    PDFIngestionError,
    UnsupportedPDFError
)


def create_pdf_helper(file_path: Path, page_texts: list[str]) -> None:
    """Helper to generate a deterministic PDF file with specified text on each page."""
    writer = pypdf.PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        if text:  # If text provided, attach stream object & font
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode("utf-8"))
            font_obj = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            fonts = DictionaryObject({NameObject("/F1"): font_obj})
            resources = DictionaryObject({NameObject("/Font"): fonts})
            page[NameObject("/Resources")] = resources
            page[NameObject("/Contents")] = writer._add_object(stream)

    with open(file_path, "wb") as f:
        writer.write(f)


class TestPDFIngestionEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_valid_text_pdf(self):
        """Test A: Valid text PDF extraction, chunk production, and metadata preservation."""
        pdf_path = self.temp_path / "valid_sample.pdf"
        page_texts = [
            "This is page 1 content of Knowledge Transfer document.",
            "This is page 2 content with detailed specifications for Phase 1."
        ]
        create_pdf_helper(pdf_path, page_texts)

        chunks = load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=150)

        self.assertGreater(len(chunks), 0, "Chunks should be produced for valid text PDF.")
        for chunk in chunks:
            self.assertTrue(chunk.page_content.strip(), "Chunk content should not be empty.")
            self.assertEqual(chunk.metadata.get("source"), "valid_sample.pdf", "Source metadata must be filename.")
            self.assertIn("page", chunk.metadata, "Page metadata must exist.")
            self.assertIsInstance(chunk.metadata["page"], int, "Page metadata must be an integer.")
            self.assertGreaterEqual(chunk.metadata["page"], 1, "Page metadata must be 1-indexed.")

    def test_chunking_configuration_and_metadata(self):
        """Test B: Verify chunk_size=1000, chunk_overlap=150, and metadata retention."""
        pdf_path = self.temp_path / "large_text.pdf"
        # Create text exceeding chunk size (e.g. 2500 characters)
        paragraph = "Knowledge transfer chunking test sentence. " * 30  # ~1300 chars
        page_texts = [paragraph, paragraph]
        create_pdf_helper(pdf_path, page_texts)

        chunks = load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=150)

        self.assertGreater(len(chunks), 1, "Long text should be split into multiple chunks.")
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 1000, "Chunk length must not exceed chunk_size=1000.")
            self.assertEqual(chunk.metadata.get("source"), "large_text.pdf")
            self.assertIn(chunk.metadata.get("page"), [1, 2])

    def test_invalid_corrupt_pdf(self):
        """Test C: Corrupted / invalid PDF raises clear PDFIngestionError."""
        corrupt_path = self.temp_path / "corrupt.pdf"
        with open(corrupt_path, "wb") as f:
            f.write(b"%PDF-1.4 invalid corrupt binary data stream without valid trailer")

        with self.assertRaises(PDFIngestionError) as ctx:
            load_and_split_pdf(corrupt_path)

        self.assertIn("Failed to parse PDF file", str(ctx.exception))

    def test_empty_pdf_zero_pages(self):
        """Test D: PDF with zero pages raises clear PDFIngestionError."""
        empty_pdf_path = self.temp_path / "empty_pages.pdf"
        writer = pypdf.PdfWriter()
        with open(empty_pdf_path, "wb") as f:
            writer.write(f)

        with self.assertRaises(PDFIngestionError) as ctx:
            load_and_split_pdf(empty_pdf_path)

        self.assertIn("contains zero pages", str(ctx.exception))

    def test_scanned_image_only_pdf(self):
        """Test E: Scanned / image-only PDF without text raises UnsupportedPDFError."""
        scanned_pdf_path = self.temp_path / "scanned_image.pdf"
        # Blank pages with no text content streams simulate scanned/image-only PDFs
        create_pdf_helper(scanned_pdf_path, ["", ""])

        with self.assertRaises(UnsupportedPDFError) as ctx:
            load_and_split_pdf(scanned_pdf_path)

        self.assertIn("contains no extractable text", str(ctx.exception))

    def test_missing_file(self):
        """Test F: Missing PDF file raises clear PDFIngestionError."""
        missing_path = self.temp_path / "does_not_exist.pdf"

        with self.assertRaises(PDFIngestionError) as ctx:
            load_and_split_pdf(missing_path)

        self.assertIn("not found", str(ctx.exception))

    def test_invalid_file_extension(self):
        """Additional Validation: Non-PDF extension raises PDFIngestionError."""
        text_file = self.temp_path / "document.txt"
        with open(text_file, "w") as f:
            f.write("Some text")

        with self.assertRaises(PDFIngestionError) as ctx:
            load_and_split_pdf(text_file)

        self.assertIn("Expected a '.pdf' file", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
