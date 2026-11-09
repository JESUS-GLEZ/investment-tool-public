"""Offline trust-store controls for the October 2026 emSign correction."""

import hashlib
from pathlib import Path
import ssl

import certifi
from cryptography import x509
from cryptography.hazmat.primitives import hashes
import httpx
import requests


COMPROMISED_ROOT = "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b"
UNAFFECTED_EMSIGN_ROOTS = {
    "40f6af0346a99aa1cd1d555a4e9cce62c7f9634603ee406615833dc8c8d00367",
    "125609aa301da0a249b97a8239cb6a34216f44dcac9f3954b14292f2e8c8608f",
    "bc4d809b15189d78db3e1d8cf4f9726a795da1643ca5f1358e1ddb0edc0d7eb3",
}


def bundle_fingerprints():
    return {
        cert.fingerprint(hashes.SHA256()).hex()
        for cert in x509.load_pem_x509_certificates(Path(certifi.where()).read_bytes())
    }


def context_fingerprints(context):
    return {
        hashlib.sha256(der).hexdigest()
        for der in context.get_ca_certs(binary_form=True)
    }


def test_compromised_emsign_ecc_g3_is_not_trusted():
    assert COMPROMISED_ROOT not in bundle_fingerprints()


def test_unaffected_emsign_roots_remain_in_bundle():
    assert UNAFFECTED_EMSIGN_ROOTS <= bundle_fingerprints()


def test_bundle_loads_with_certificate_and_hostname_verification():
    context = ssl.create_default_context(cafile=certifi.where())
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    assert UNAFFECTED_EMSIGN_ROOTS <= context_fingerprints(context)
    assert COMPROMISED_ROOT not in context_fingerprints(context)


def test_requests_default_bundle_uses_corrected_certifi():
    assert Path(requests.certs.where()).samefile(certifi.where())
    assert Path(requests.utils.DEFAULT_CA_BUNDLE_PATH).samefile(certifi.where())


def test_httpx_default_context_keeps_verification_and_corrected_roots():
    context = httpx.create_ssl_context(verify=True, trust_env=False)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    assert UNAFFECTED_EMSIGN_ROOTS <= context_fingerprints(context)
    assert COMPROMISED_ROOT not in context_fingerprints(context)
