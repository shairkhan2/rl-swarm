import { TurnkeyClient } from "@turnkey/http";
import { ApiKeyStamper } from "@turnkey/api-key-stamper";
import { getLatestApiKey, getUser } from "@/app/db";
import { NextResponse } from "next/server";
import contract from "@/app/lib/contract.json";
import {
  Address,
  createWalletClient,
  Hex,
  SignableMessage,
  hashMessage,
  encodeFunctionData,
  decodeErrorResult,
} from "viem";
import {
  alchemy,
  createAlchemySmartAccountClient,
  gensynTestnet,
} from "@account-kit/infra";
import { toAccount } from "viem/accounts";
import { WalletClientSigner } from "@aa-sdk/core";
import { createModularAccountV2 } from "@account-kit/smart-contracts";
import { SendUserOperationErrorType } from "viem/account-abstraction";
import { httpRequestErroDetailsStringSchema } from "@/app/lib/HttpRequestError";

const TURNKEY_BASE_URL = "https://api.turnkey.com";
const ALCHEMY_BASE_URL = "https://api.g.alchemy.com";

export async function POST(request: Request) {
  const body: { orgId: string; peerId: string } = await request
    .json()
    .catch((err) => {
      console.error(err);
      return NextResponse.json(
        { error: "bad request" },
        {
          status: 400,
        },
      );
    });
  if (!body.orgId) {
    return NextResponse.json(
      { error: "bad request" },
      {
        status: 400,
      },
    );
  }
  console.log(body.orgId);

  try {
    const user = getUser(body.orgId);
    if (!user) {
      return NextResponse.json(
        { error: "user not found" },
        {
          status: 404,
        },
      );
    }
    const apiKey = getLatestApiKey(body.orgId);
    if (!apiKey || !apiKey.deferredActionDigest) {
      return NextResponse.json(
        { error: "api key not found" },
        {
          status: 500,
        },
      );
    }
    const transport = alchemy({
      apiKey: process.env.NEXT_PUBLIC_ALCHEMY_API_KEY!,
    });

    const walletClient = createWalletClient({
      account: user.address as `0x${string}`,
      chain: gensynTestnet,
      transport,
    });

    const account = await createModularAccountV2({
      transport,
      chain: gensynTestnet,
      signer: new WalletClientSigner(walletClient, "wallet"),
      deferredAction: apiKey.deferredActionDigest as `0x${string}`,
    });

    const client = createAlchemySmartAccountClient({
      account,
      chain: gensynTestnet,
      transport,
      policyId: process.env.NEXT_PUBLIC_PAYMASTER_POLICY_ID!,
    });

    // Check if the user's address already registered for better error handling.
    /*
    const existingPeerId = await client.readContract({
      abi: [
        {
          inputs: [
            {
              internalType: "address",
              name: "eoa",
              type: "address",
            },
          ],
          name: "getPeerId",
          outputs: [
            {
              internalType: "string",
              name: "",
              type: "string",
            },
          ],
          stateMutability: "view",
          type: "function",
        },
      ],
      functionName: "getPeerId",
      args: [account.address as Address],
      address: "0x6484a07281B72b8b541A86Ec055534223672c2fb",
    });
    if (existingPeerId) {
      console.log(
        `Address ${account.address} already registered with peerId ${existingPeerId}`,
      );
      return NextResponse.json(
        { error: "account address already registered" },
        {
          status: 400,
        },
      );
    }
    */

    const contractAdrr = process.env.SMART_CONTRACT_ADDRESS! as `0x${string}`;
    console.log(contractAdrr);

    const { hash } = await client.sendUserOperation({
      uo: {
        target: contractAdrr,
        data: encodeFunctionData({
          abi: contract.abi,
          functionName: "registerPeer",
          args: [body.peerId],
        }),
      },
    });

    return NextResponse.json(
      {
        hash,
      },
      {
        status: 200,
      },
    );
  } catch (err) {
    console.error(err);
    // Casting is not ideal but is canonical way of handling errors as per the
    // viem docs.
    //
    // See: https://viem.sh/docs/error-handling#error-handling
    const error = err as SendUserOperationErrorType;
    if (error.name !== "HttpRequestError") {
      return NextResponse.json(
        {
          error: "An unexpected error occurred",
          original: error,
        },
        {
          status: 500,
        },
      );
    }

    const parsedDetailsResult = httpRequestErroDetailsStringSchema.safeParse(
      error.details,
    );

    if (!parsedDetailsResult.success) {
      return NextResponse.json(
        {
          error: "An unexpected error occurred getting request details",
          parseError: parsedDetailsResult.error,
          original: error.details,
        },
        {
          status: 500,
        },
      );
    }

    const {
      data: {
        data: { revertData },
      },
    } = parsedDetailsResult;

    const decodedError = decodeErrorResult({
      data: revertData,
      abi: contract.abi,
    });

    return NextResponse.json(
      {
        error: decodedError.errorName,
        metaMessages: error.metaMessages,
      },
      {
        status: 400,
      },
    );
  }
}
