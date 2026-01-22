"""Example integration test to verify test infrastructure."""

import pytest


@pytest.mark.integration
def test_integration_example():
    """Test integration example."""
    string = "integration_test_example"
    assert string == "integration_test_example"


@pytest.mark.integration
def test_multi_component_integration(sample_documents, sample_metadata):
    """Test integration of multiple components."""
    # Simulate processing multiple documents
    processed = []
    for doc in sample_documents:
        processed.append({
            "content": doc,
            "metadata": sample_metadata,
        })
    
    assert len(processed) == len(sample_documents)
    assert all("content" in item for item in processed)
    assert all("metadata" in item for item in processed)


@pytest.mark.integration
@pytest.mark.slow
def test_slow_integration():
    """Test that takes longer to run."""
    # Simulate slow operation
    import time
    time.sleep(0.1)
    
    result = sum(range(1000))
    assert result == 499500


@pytest.mark.integration
class TestIntegrationSuite:
    """Integration test suite."""

    def test_component_a_to_b(self):
        """Test flow from component A to B."""
        data = {"input": "test"}
        
        # Simulate component A processing
        processed = f"{data['input']}_processed"
        
        # Simulate component B receiving
        assert "_processed" in processed

    def test_component_b_to_c(self):
        """Test flow from component B to C."""
        data = "test_processed"
        
        # Simulate component B output
        output_b = data.upper()
        
        # Simulate component C receiving
        final = f"{output_b}_final"
        assert "TEST_PROCESSED_FINAL" == final

    def test_end_to_end_flow(self):
        """Test complete end-to-end flow."""
        # Input
        input_data = "start"
        
        # Component A
        step1 = f"{input_data}_a"
        
        # Component B
        step2 = f"{step1}_b"
        
        # Component C
        step3 = f"{step2}_c"
        
        assert step3 == "start_a_b_c"
