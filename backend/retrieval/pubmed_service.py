import requests
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import time

logger = logging.getLogger(__name__)

class PubMedService:
    """
    Service to interact with NCBI PubMed API (E-utilities).
    Strictly follows 'No web scraping' rule.
    """
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, email: str = "researcher@example.com", tool: str = "medical_reasoning_agent"):
        self.params = {
            "email": email,
            "tool": tool,
            "db": "pubmed"
        }

    def search(self, query: str, retmax: int = 10) -> List[str]:
        """
        Uses esearch to find PMIDs for a query.
        """
        url = self.BASE_URL + "esearch.fcgi"
        params = {
            **self.params,
            "term": query,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance"
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            return id_list
        except Exception as e:
            logger.error(f"PubMed search failed for '{query}': {e}")
            return []

    def fetch_details(self, pmids: List[str]) -> List[Dict]:
        """
        Uses efetch to retrieve metadata (Title, Abstract, Year, MeSH) for PMIDs.
        """
        if not pmids:
            return []

        url = self.BASE_URL + "efetch.fcgi"
        ids_str = ",".join(pmids)
        params = {
            **self.params,
            "id": ids_str,
            "retmode": "xml"
        }

        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            return self._parse_pubmed_xml(response.content)
        except Exception as e:
            logger.error(f"PubMed fetch details failed: {e}")
            return []

    def _parse_pubmed_xml(self, xml_content: bytes) -> List[Dict]:
        """
        Parses the PubMed XML response.
        """
        articles = []
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall(".//PubmedArticle"):
                medline_citation = article.find("MedlineCitation")
                article_data = medline_citation.find("Article")
                
                # PMID
                pmid = medline_citation.find("PMID").text
                
                # Title
                title = article_data.find("ArticleTitle").text
                
                # Abstract
                abstract_text = ""
                abstract = article_data.find("Abstract")
                if abstract is not None:
                    abstract_parts = []
                    for text in abstract.findall("AbstractText"):
                        if text.text:
                            label = text.get("Label", "")
                            content = text.text
                            if label:
                                abstract_parts.append(f"{label}: {content}")
                            else:
                                abstract_parts.append(content)
                    abstract_text = "\n".join(abstract_parts)
                
                # Year
                pub_date = article_data.find("Journal/JournalIssue/PubDate")
                year = "Unknown"
                if pub_date is not None:
                    year_node = pub_date.find("Year")
                    if year_node is not None:
                        year = year_node.text
                    else:
                        # Sometimes MedlineDate is used
                        medline_date = pub_date.find("MedlineDate")
                        if medline_date is not None:
                            year = medline_date.text.split(" ")[0]

                # MeSH Terms
                mesh_terms = []
                mesh_heading_list = medline_citation.find("MeshHeadingList")
                if mesh_heading_list is not None:
                    for mesh in mesh_heading_list.findall("MeshHeading"):
                        descriptor = mesh.find("DescriptorName")
                        if descriptor is not None:
                            mesh_terms.append(descriptor.text)
                
                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract_text,
                    "year": year,
                    "mesh": mesh_terms
                })
                
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            
        return articles

    def search_and_fetch(self, query: str, retmax: int = 5) -> List[Dict]:
        ids = self.search(query, retmax)
        if ids:
            return self.fetch_details(ids)
        return []
