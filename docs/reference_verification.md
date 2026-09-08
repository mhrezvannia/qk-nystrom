# Reference verification (3 Sep 2026)

Method: each record checked against the publisher page, arXiv abstract page, ADS/OSTI record or OpenReview (via web fetch/search in this session). "Publisher" = confirmed on the publisher/arXiv page; "Index" = confirmed via ADS/OSTI/DOAJ index; "Standard" = canonical, widely cited record with DOI, not re-fetched in this session (flag for the authors' final check if desired).

| Reference (as now cited) | Verified? | Authoritative source | DOI / arXiv | Correct in previous draft? | Action |
|---|---|---|---|---|---|
| Coelho, Kruse, Rosskopf (2025). Quantum-efficient kernel target alignment. Proc. ICAART 2025, Vol. 1, SciTePress, 763–772 | Yes | SciTePress PDF (title page: authors, affiliations Fraunhofer IISB / TU Munich, pages, DOI, ISBN 978-989-758-737-5) | 10.5220/0013391500003890; arXiv:2502.08225 | No — cited as "Anonymous (2025)" / "Quantum-Efficient KTA 2025" | Completed; in-text citations updated (§1, §2.4, §6, related_work.md) |
| Siva, T. (2025). Nyström quantum kernel estimation for efficient environmental monitoring. IEEE Access 13, 10331–10342 | **No** | Not found: IEEE Xplore/Google-indexed searches on exact title (2 by the authors + 2 in this session) return nothing | — | Unverifiable (inherited from the original manuscript's ref. [7]) | **Deleted** from reference list, §1, §2.4 and related_work.md; sentences reworded to rest on Srikumar 2022, Coelho 2025, Rezvannia 2026 |
| Ragone et al. Representation theory for geometric quantum machine learning | Yes (arXiv) | arXiv abstract 2210.07980; the PRX Quantum 5, 020328 record previously attached to it belongs to **Nguyen et al., "Theory for equivariant quantum neural networks"** (APS page fetched) | arXiv:2210.07980 | **No** — wrong venue/article number (inherited from the original manuscript's ref. [2]) | Corrected to arXiv; Nguyen et al. 2024 added as a separate verified reference where equivariant-QNN theory is cited |
| Nguyen, Schatzki, Braccia, Ragone, Coles, Sauvage, Larocca, Cerezo (2024). Theory for equivariant quantum neural networks. PRX Quantum 5, 020328 | Yes | APS abstract page (authors, volume, article, date 6 May 2024) | 10.1103/PRXQuantum.5.020328 | — (new) | Added |
| Srikumar, Hill, Hollenberg (2023). A kernel-based quantum random forest for improved classification. arXiv:2210.02355 v2 | Yes (arXiv) | arXiv abstract page: authors, v1 5 Oct 2022, v2 19 Feb 2023, no journal-ref | arXiv:2210.02355 | No — draft claimed "/ Quantum Machine Intelligence 2024" without evidence | Journal claim removed; cited as arXiv preprint |
| Schnabel, Roth (2025). Quantum kernel methods under scrutiny: a benchmarking study. QMI 7, 58 | Yes | Springer article page (published 24 Apr 2025) | 10.1007/s42484-025-00273-5; arXiv:2409.04406 | Partially (article number missing) | Article number and DOI added |
| Glick et al. (2024). Covariant quantum kernels for data with group structure. Nat. Phys. 20, 479–483 | Yes | Nature Physics page (published 19 Jan 2024) | 10.1038/s41567-023-02340-9; arXiv:2105.03406 | Yes | DOI added |
| Thanasilp, Wang, Cerezo, Holmes (2024). Exponential concentration in quantum kernel methods. Nat. Commun. 15, 5200 | Yes | Nature page + ADS 2024NatCo..15.5200T | 10.1038/s41467-024-49287-w | Yes | DOI added |
| Wiersema, Kemper, Bakalov, Killoran (2025). Geometric QML with horizontal quantum gates. PRR 7, 013148 | Yes | APS link.aps.org record; arXiv:2406.04418 | 10.1103/PhysRevResearch.7.013148 | Yes | DOI added |
| Schatzki, Larocca, Nguyen, Sauvage, Cerezo (2024). Theoretical guarantees for permutation-equivariant QNNs. npj QI 10, 12 | Yes | Nature page + ADS 2024npjQI..10...12S | 10.1038/s41534-024-00804-1 | Yes | DOI added |
| Meyer et al. (2023). Exploiting symmetry in variational QML. PRX Quantum 4, 010328 | Yes | APS abstract page (DOAJ Mar 2023) | 10.1103/PRXQuantum.4.010328 | Yes | DOI added |
| Larocca et al. (2022). Group-invariant quantum machine learning. PRX Quantum 3, 030341 | Yes | APS abstract page; OSTI 1887927 | 10.1103/PRXQuantum.3.030341 | Yes | DOI added |
| Canatar, Peters, Pehlevan, Wild, Shaydulin (2023). Bandwidth enables generalization in quantum kernel models. TMLR | Yes | OpenReview forum A1N2qp4yAq (TMLR); arXiv:2206.06686 | OpenReview A1N2qp4yAq | Yes (venue abbreviation only) | Venue spelled out, identifiers added |
| Havlíček et al. (2019). Nature 567, 209–212 | Standard | — | 10.1038/s41586-019-0980-2 | Yes | DOI added |
| Schuld, Killoran (2019). PRL 122, 040504 | Standard | — | 10.1103/PhysRevLett.122.040504 | Yes | DOI added |
| Schuld (2021). Supervised quantum machine learning models are kernel methods. arXiv:2101.11020 | Standard | — | arXiv:2101.11020 | Yes | — |
| Schuld, Sweke, Meyer (2021). PRA 103, 032430 | Standard | — | 10.1103/PhysRevA.103.032430 | Yes | DOI added |
| Shaydulin, Wild (2022). PRA 106, 042407 | Standard | — | 10.1103/PhysRevA.106.042407 | Yes | DOI added |
| Huang et al. (2021). Power of data in QML. Nat. Commun. 12, 2631 | Standard | — | 10.1038/s41467-021-22539-9 | Yes | DOI added |
| Williams, Seeger (2001). Using the Nyström method to speed up kernel machines. NIPS 13, 682–688 | Standard | — | — | Yes | — |
| Drineas, Mahoney (2005). JMLR 6, 2153–2175 | Standard | — | — | Yes | — |
| Kumar, Mohri, Talwalkar (2012). Sampling methods for the Nyström method. JMLR 13, 981–1006 | Standard | — | — | Yes | — |
| Gittens, Mahoney (2016). Revisiting the Nyström method… JMLR 17(117), 1–65 | Standard | — | — | Yes (issue number added) | Issue added |
| Zhang, Tsang, Kwok (2008). Improved Nyström low-rank approximation and error analysis. ICML, 1232–1239 | Standard | — | 10.1145/1390156.1390311 | Yes | — |
| Oglic, Gärtner (2017). Nyström method with kernel k-means++ samples as landmarks. ICML, PMLR 70, 2652–2660 | Standard | PMLR page (fetched in the first review session) | — | Yes | — |
| Rahimi, Recht (2007). Random features for large-scale kernel machines. NIPS 20 | Standard | — | — | Yes | — |
| Rezvannia (2026). Hybrid QML framework based on Nyström approximation… Research Square rs-8474206/v1 | Yes | Research Square page (fetched in the first review session: author, date 18 Feb 2026, dataset) | rs-8474206/v1 | Yes | — |

**Claim–source consistency check (Task 1c).** Every sentence that cites a reference was re-read against the record's abstract: Coelho et al. is cited for Nyström + KTA + noise experiments (their abstract states coherent and depolarising noise — the manuscript sentence was adjusted to say so); Srikumar et al. for Nyström inside a quantum random forest with generalisation bounds (abstract confirms); Glick et al. for covariant kernels with fiducial-state optimisation and hardware demonstration (confirms); Nguyen et al. is now cited only in the list of equivariant/invariant variational-QML works. No other mismatch found.

**Removed/absent:** Siva 2025 (unverifiable). No reference is retained "because it appeared in an earlier draft".
