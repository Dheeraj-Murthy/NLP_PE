import os
from typing import List, Optional, Dict, Any
from pathlib import Path


class DocumentProcessor:
    SUPPORTED_IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
    SUPPORTED_DOCUMENT_FORMATS = {".pdf"}

    def __init__(self):
        self.pytesseract_available = False
        self.pdf2image_available = False
        self._check_dependencies()

    def _check_dependencies(self):
        try:
            import pytesseract

            self.pytesseract_available = True
        except ImportError:
            pass

        try:
            from pdf2image import convert_from_path

            self.pdf2image_available = True
        except ImportError:
            pass

    def process_document(
        self,
        document_path: str,
        extract_only: bool = False,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = Path(document_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {document_path}",
                "text": "",
                "source": str(path),
            }

        ext = path.suffix.lower()

        if ext in self.SUPPORTED_IMAGE_FORMATS:
            return self._process_image(path, extract_only, query)
        elif ext in self.SUPPORTED_DOCUMENT_FORMATS:
            return self._process_pdf(path, extract_only, query)
        else:
            return {
                "success": False,
                "error": f"Unsupported file format: {ext}",
                "text": "",
                "source": str(path),
            }

    def _process_image(
        self, path: Path, extract_only: bool, query: Optional[str]
    ) -> Dict[str, Any]:
        if not self.pytesseract_available:
            return {
                "success": False,
                "error": "pytesseract not installed. Run: pip install pytesseract",
                "text": "",
                "source": str(path),
            }

        try:
            import pytesseract
            from PIL import Image

            image = Image.open(path)
            text = pytesseract.image_to_string(image)

            return {
                "success": True,
                "text": text.strip(),
                "source": str(path),
                "pages": 1,
                "document_type": "image",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"OCR processing failed: {str(e)}",
                "text": "",
                "source": str(path),
            }

    def _process_pdf(
        self, path: Path, extract_only: bool, query: Optional[str]
    ) -> Dict[str, Any]:
        if not self.pdf2image_available or not self.pytesseract_available:
            return {
                "success": False,
                "error": "pdf2image/pytesseract not installed. Run: pip install pdf2image pytesseract poppler",
                "text": "",
                "source": str(path),
            }

        try:
            from pdf2image import convert_from_path
            import pytesseract
            from PIL import Image

            images = convert_from_path(str(path))
            all_text = []
            page_count = len(images)

            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image)
                all_text.append(f"--- Page {i + 1} ---\n{page_text}")

            full_text = "\n\n".join(all_text)

            return {
                "success": True,
                "text": full_text.strip(),
                "source": str(path),
                "pages": page_count,
                "document_type": "pdf",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"PDF OCR processing failed: {str(e)}",
                "text": "",
                "source": str(path),
            }

    def get_status(self) -> Dict[str, bool]:
        return {
            "pytesseract": self.pytesseract_available,
            "pdf2image": self.pdf2image_available,
            "ready": self.pytesseract_available,
        }


if __name__ == "__main__":
    processor = DocumentProcessor()
    status = processor.get_status()
    print("Document Processor Status:")
    print(f"  OCR (pytesseract): {'✓' if status['pytesseract'] else '✗'}")
    print(f"  PDF (pdf2image): {'✓' if status['pdf2image'] else '✗'}")
    print(f"  Ready: {'✓' if status['ready'] else '✗'}")
