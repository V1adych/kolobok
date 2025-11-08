from dataclasses import dataclass
from PIL import Image
from pathlib import Path
import base64
import io
import time
import asyncio
import aiohttp
import statistics
from typing import List

import numpy as np
import tyro


@dataclass
class Args:
    num_requests: int = 3
    max_rps: int = 10
    images_dir: str = "perf/data/thread"
    url: str = "http://localhost:8000/api/v1/analyze_thread"
    token: str = ""


@dataclass
class RequestResult:
    success: bool
    response_time: float
    status_code: int
    error: str = ""
    response_text: str = ""


def get_image_base64(image_path: str) -> str:
    img = Image.open(image_path)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


async def test_request(session: aiohttp.ClientSession, url: str, payload: dict, headers: dict = None) -> RequestResult:
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers) as response:
            end = time.perf_counter()
            response_time = end - start
            success = response.status == 200
            response_text = await response.text()
            return RequestResult(
                success=success,
                response_time=response_time,
                status_code=response.status,
                response_text=response_text,
            )
    except Exception as e:
        end = time.perf_counter()
        return RequestResult(success=False, response_time=end - start, status_code=0, error=str(e))


async def rate_limited_requests(
    session: aiohttp.ClientSession,
    url: str,
    payloads: List[dict],
    max_rps: int,
    headers: dict = None,
) -> List[RequestResult]:
    """Execute requests with rate limiting"""
    results = []
    semaphore = asyncio.Semaphore(max_rps)

    async def limited_request(payload: dict) -> RequestResult:
        async with semaphore:
            result = await test_request(session, url, payload, headers)
            # Rate limiting: ensure we don't exceed max_rps
            await asyncio.sleep(1.0 / max_rps)
            return result

    # Create tasks for all requests
    tasks = [limited_request(payload) for payload in payloads]

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)

    return results


def print_statistics(results: List[RequestResult], total_time: float):
    """Print comprehensive statistics about the test run"""
    total_requests = len(results)
    successful_requests = sum(1 for r in results if r.success)
    failed_requests = total_requests - successful_requests

    success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0

    # Response time statistics for successful requests
    successful_times = [r.response_time for r in results if r.success]

    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)
    print(f"Total requests:        {total_requests}")
    print(f"Successful requests:   {successful_requests}")
    print(f"Failed requests:       {failed_requests}")
    print(f"Success rate:          {success_rate:.2f}%")
    print(f"Total test time:       {total_time:.2f}s")
    print(f"Actual RPS:            {total_requests / total_time:.2f}")

    if successful_times:
        print("\nResponse Time Statistics (successful requests):")
        print(f"Min response time:     {min(successful_times):.3f}s")
        print(f"Max response time:     {max(successful_times):.3f}s")
        print(f"Avg response time:     {statistics.mean(successful_times):.3f}s")
        print(f"Median response time:  {statistics.median(successful_times):.3f}s")
        if len(successful_times) > 1:
            print(f"Std dev response time: {statistics.stdev(successful_times):.3f}s")

    # Error statistics
    error_codes = {}
    error_messages = {}
    failed_requests_details = []

    for i, result in enumerate(results):
        if not result.success:
            error_codes[result.status_code] = error_codes.get(result.status_code, 0) + 1
            if result.error:
                error_messages[result.error] = error_messages.get(result.error, 0) + 1
            failed_requests_details.append((i + 1, result))

    if error_codes:
        print("\nError Status Codes:")
        for code, count in sorted(error_codes.items()):
            print(f"  {code}: {count} occurrences")

    if error_messages:
        print("\nError Messages:")
        for error, count in sorted(error_messages.items()):
            print(f"  {error}: {count} occurrences")

    # Print detailed failed request information
    if failed_requests_details:
        print("\nFailed Request Details:")
        print("-" * 60)
        for request_num, result in failed_requests_details:
            print(f"Request #{request_num}:")
            print(f"  Status Code: {result.status_code}")
            print(f"  Response Time: {result.response_time:.3f}s")
            if result.error:
                print(f"  Error: {result.error}")
            if result.response_text:
                print(f"  Response: {result.response_text[:500]}{'...' if len(result.response_text) > 500 else ''}")
            print("-" * 60)

    print("=" * 60)


async def main():
    args = tyro.cli(Args)

    # Get all image paths
    all_image_paths = list(map(str, Path(args.images_dir).glob("*.png")))
    if not all_image_paths:
        print(f"No images found in {args.images_dir}")
        return

    print(f"Found {len(all_image_paths)} images in {args.images_dir}")
    # Load all images to RAM as base64 first
    print("Loading all images to RAM...")
    image_payloads = []
    for image_path in all_image_paths:
        try:
            image_base64 = get_image_base64(image_path)
            image_payloads.append({"image": image_base64})
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    if not image_payloads:
        print("No valid images loaded")
        return

    print(f"Loaded {len(image_payloads)} images to RAM")

    # Select random payloads for requests
    payloads = list(np.random.choice(image_payloads, args.num_requests, replace=True))

    print(f"Preparing {args.num_requests} requests with max {args.max_rps} RPS...")

    # Prepare headers
    headers = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    print("Starting load test...")
    start_time = time.perf_counter()

    # Execute requests
    async with aiohttp.ClientSession() as session:
        results = await rate_limited_requests(session, args.url, payloads, args.max_rps, headers)

    end_time = time.perf_counter()
    total_time = end_time - start_time

    # Print statistics
    print_statistics(results, total_time)


if __name__ == "__main__":
    asyncio.run(main())
